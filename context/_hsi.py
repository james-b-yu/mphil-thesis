import os
from typing import Any, Mapping
import torch
from tqdm import tqdm
from ._noise import Noise
from ._interp import Interpolate
from ._sampler import Sampler
from config import Config
from argparse import Namespace
from logging import Logger
from datasets import get_dataset
from model import FNO
from torch.utils.data import DataLoader
from torch.optim import AdamW
from time import time
from wandb import init as wandb_init


class HilbertStochasticInterpolant:
    def __init__(self, args: Namespace, config: Config, logger: Logger):
        self.args = args
        self.config = config
        self.logger = logger
        self.device = config["device"]
        self.noise = Noise(config)
        self.interpolate = Interpolate(config)
        self.device = config["device"]
        self.dimension = config["dimension"]
        self.sampler = Sampler(config)

    def train(self):
        wandb_run = wandb_init()
        self.logger.info("training")

        train_dataset = get_dataset(
            self.config["data"]["dataset"], phase="train")
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["training"]["n_batch"], shuffle=True)

        model_forward = FNO(self.config)
        model_backward = FNO(self.config)

        if self.args.resume is not None:
            state_dict = torch.load(self.args.resume)
            model_forward.load_state_dict(state_dict["forward"])
            model_backward.load_state_dict(state_dict["backward"])

        model_forward = model_forward.to(device=self.device)
        model_backward = model_backward.to(device=self.device)

        optim_forward = AdamW(model_forward.parameters(), amsgrad=True,
                              lr=self.config["training"]["lr"])
        optim_backward = AdamW(model_backward.parameters(), amsgrad=True,
                               lr=self.config["training"]["lr"])
        step = 0

        for epoch in range(self.args.start_epoch, self.config["training"]
                           ["n_epochs"] + 1):
            data_start = time()
            data_time = 0

            for i, (x0, x1) in enumerate(train_loader):
                model_forward.train()

                step += 1
                x0 = x0.to(self.device, dtype=torch.float32).squeeze(-1)
                x1 = x1.to(self.device, dtype=torch.float32).squeeze(-1)

                data_time += time() - data_start

                t = torch.rand(x1.shape[0], device=self.device)
                z = self.noise.sample(x1.shape)

                xt = self.interpolate(t, x0, x1, z)
                target_forward = self.interpolate.get_target_forward_drift(
                    t, x0, x1, z)
                target_backward = self.interpolate.get_target_backward_drift(
                    t, x0, x1, z)

                pred_forward = model_forward(xt, t)
                pred_backward = model_backward(xt, t)

                mse_forward = (target_forward -
                               # TODO: accommodate 2D etc
                               pred_forward).square().sum(dim=(1, 2)).mean(dim=0)
                mse_backward = (target_backward -
                                # TODO: accommodate 2D etc
                                pred_backward).square().sum(dim=(1, 2)).mean(dim=0)

                if torch.isnan(mse_forward) or torch.isnan(mse_backward):
                    self.logger.warning("encountered NAN loss. skipping")
                    continue

                optim_forward.zero_grad()
                optim_backward.zero_grad()

                mse_forward.backward()
                mse_backward.backward()

                norm_forward = torch.nn.utils.clip_grad_norm_(
                    model_forward.parameters(
                    ), self.config["training"]["grad_clip"]
                )
                norm_backward = torch.nn.utils.clip_grad_norm_(
                    model_backward.parameters(
                    ), self.config["training"]["grad_clip"]
                )
                optim_forward.step()
                optim_backward.step()

                self.logger.info(
                    f"step: {step}, epoch: {epoch} forward: {torch.abs(mse_forward).item():.2f} backward: {torch.abs(mse_backward).item():.2f}, data time: {data_time / (i+1)}"
                )
                wandb_run.log({
                    "step": step,
                    "forward_loss": mse_forward,
                    "backward_loss": mse_backward,
                    "forward_gradnorm": norm_forward,
                    "backward_gradnorm": norm_backward,
                    "epoch": epoch
                })

                if self.args.save_every is not None and epoch % self.args.save_every == 0 and epoch > 1:
                    os.makedirs(self.args.save_dir, exist_ok=True)
                    torch.save({
                        "forward": model_forward.state_dict(),
                        "backward": model_backward.state_dict(),
                    }, f"{self.args.save_dir}/epoch_{epoch}.pth")

                data_start = time()

    def test(self, state_dict: Mapping[str, Any], max_n_samples: int, n_batch_size: int, all_t: bool):
        self.logger.info("testing")

        model_forward = FNO(self.config)
        model_forward.load_state_dict(state_dict["forward"])
        model_forward = model_forward.to(device=self.device)
        model_forward.eval()

        times = torch.linspace(
            self.config["sampling"]["start_t"], self.config["sampling"]["end_t"], self.config["sampling"]["n_t_steps"])

        test_dataset = get_dataset(
            self.config["data"]["dataset"], phase="test")
        n_samples = min(max_n_samples, len(test_dataset)
                        ) if max_n_samples is not None else len(test_dataset)
        test_loader = DataLoader(
            test_dataset, batch_size=n_batch_size, shuffle=False)

        res_forward = torch.zeros(size=(len(times), n_samples, self.dimension)
                                  if all_t else (n_samples, self.dimension))

        start_cur = 0
        mses_forward = []

        for i, (x0, x1) in enumerate(tqdm(test_loader)):
            if start_cur >= n_samples:
                break

            n_batch_size = x0.shape[0]
            if start_cur + n_batch_size > n_samples:
                n_batch_size = n_samples - start_cur
                x0 = x0[:n_batch_size]
                x1 = x1[:n_batch_size]
            end_cur = start_cur + n_batch_size

            x0 = x0.to(device=self.device, dtype=torch.float32)
            x1 = x1.to(device=self.device, dtype=torch.float32)

            X_forward = self.sampler(
                x0, model_forward, self.noise, times, all_t)

            if all_t:
                res_forward[:, start_cur:end_cur] = X_forward[:, :, 1]
            else:
                res_forward[start_cur:end_cur] = X_forward[:, 1]

            start_cur = end_cur

            # now perform evaluation
            if all_t:
                X_forward = X_forward[-1]
                # X_backward = X_backward[-1]

            mses_forward.append(((X_forward[:, 1] - x1[:, 1]) **
                                 2).sum(dim=1).mean(dim=0))

        mse_forward = torch.as_tensor(mses_forward).mean().item()

        return res_forward, mse_forward

    def test_one(self, state_dict, n_id, n_repeats, all_t):
        self.logger.info("testing one")

        model_forward = FNO(self.config)
        model_forward.load_state_dict(
            state_dict["forward"] if "forward" in state_dict else state_dict)
        model_forward = model_forward.to(device=self.device)
        model_forward.eval()

        times = torch.linspace(
            self.config["sampling"]["start_t"], self.config["sampling"]["end_t"], self.config["sampling"]["n_t_steps"])

        test_dataset = get_dataset(
            self.config["data"]["dataset"], phase="test")

        x0, x1 = test_dataset[n_id]

        x0 = x0.to(device=self.device, dtype=torch.float32)
        x1 = x1.to(device=self.device, dtype=torch.float32)

        x0 = x0[None].expand([n_repeats, x0.shape[0], x0.shape[1]])
        x1 = x1[None].expand([n_repeats, x1.shape[0], x1.shape[1]])

        X_forward = self.sampler(x0, model_forward, self.noise, times, all_t)

        model_backward = FNO(self.config)
        model_backward.load_state_dict(
            state_dict["backward"] if "backward" in state_dict else state_dict)
        model_backward = model_backward.to(device=self.device)
        model_backward.eval()

        X_backward = self.sampler(
            x1, model_backward, self.noise, times, all_t, backward=True)

        return X_forward, X_backward
