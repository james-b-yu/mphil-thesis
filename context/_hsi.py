import os
from typing import Any, Literal, Mapping
import torch
from tqdm import tqdm
from ._noise import Noise
from ._interp import Interpolate
from ._sampler import Sampler
from config import Config
from argparse import Namespace
from logging import Logger
from my_datasets import get_dataset
from model import FNO
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from time import time
from wandb import init as wandb_init
import math


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
        self.mode = config["model"]["mode"]

    def train(self):
        wandb_run = wandb_init()
        self.logger.info("training")

        train_dataset = get_dataset(
            self.config["data"]["dataset"], phase="train")
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["training"]["n_batch"], shuffle=True)

        n_warmup_steps = self.config["training"]["n_warmup_steps"]
        n_cosine_cycle_steps = self.config["training"]["n_cosine_cycle_steps"]
        max_lr = self.config["training"]["lr"]
        step = 0

        def fractional(x): return x - math.floor(x)

        def lr_function(step: int):
            if step >= n_warmup_steps:
                return 0.5 * (1 + math.cos(math.pi * fractional((step - n_warmup_steps) / n_cosine_cycle_steps)))
            else:
                return (step / n_warmup_steps)

        model_0 = FNO(self.config)  # forward if direct, EIt if separate
        model_1 = FNO(self.config)  # backward if direct, Ez if separate

        if self.args.resume is not None:
            state_dict = torch.load(self.args.resume)
            model_0.load_state_dict(
                state_dict["forward" if self.mode == "direct" else "EIt"])
            model_1.load_state_dict(
                state_dict["backward" if self.mode == "direct" else "Ez"])

        model_0 = model_0.to(device=self.device)
        model_1 = model_1.to(device=self.device)

        optim_0 = AdamW(model_0.parameters(), amsgrad=True,
                        lr=max_lr)
        optim_1 = AdamW(model_1.parameters(), amsgrad=True,
                        lr=max_lr)

        scheduler_0 = LambdaLR(optim_0, lr_function)
        scheduler_1 = LambdaLR(optim_1, lr_function)

        for epoch in range(self.args.start_epoch, self.config["training"]
                           ["n_epochs"] + 1):
            data_start = time()
            data_time = 0

            for i, (x0, x1) in enumerate(train_loader):
                model_0.train()
                model_1.train()

                step += 1
                x0 = x0.to(self.device, dtype=torch.float32).squeeze(-1)
                x1 = x1.to(self.device, dtype=torch.float32).squeeze(-1)

                data_time += time() - data_start

                t = torch.rand(x1.shape[0], device=self.device)
                z = self.noise.sample(x1.shape)

                xt = self.interpolate(t, x0, x1, z)

                if self.mode == "direct":
                    target_0 = self.interpolate.get_target_forward_drift(
                        t, x0, x1, z)
                    target_1 = self.interpolate.get_target_backward_drift(
                        t, x0, x1, z)
                elif self.mode == "separate":
                    target_0 = self.interpolate.get_target_EIt(t, x0, x1, z)
                    target_1 = self.interpolate.get_target_Ez(t, x0, x1, z)
                else:
                    raise ValueError()

                pred_0 = model_0(xt, t)
                pred_1 = model_1(xt, t)

                mse_0 = (target_0 -
                         # TODO: accommodate 2D etc
                         pred_0).square().sum(dim=(1, 2)).mean(dim=0)
                mse_1 = (target_1 -
                         # TODO: accommodate 2D etc
                         pred_1).square().sum(dim=(1, 2)).mean(dim=0)

                if torch.isnan(mse_0) or torch.isnan(mse_1):
                    self.logger.warning("encountered NAN loss. skipping")
                    continue

                # optim

                optim_0.zero_grad()
                optim_1.zero_grad()

                mse_0.backward()
                mse_1.backward()

                norm_0 = torch.nn.utils.clip_grad_norm_(
                    model_0.parameters(
                    ), self.config["training"]["grad_clip"]
                )
                norm_1 = torch.nn.utils.clip_grad_norm_(
                    model_1.parameters(
                    ), self.config["training"]["grad_clip"]
                )
                optim_0.step()
                optim_1.step()
                scheduler_0.step()
                scheduler_1.step()

                self.logger.info(
                    f"step: {step}, lr: {scheduler_0.get_last_lr()[0]:.2e} epoch: {epoch} forward: {torch.abs(mse_0).item():.2f} backward: {torch.abs(mse_1).item():.2f}, data time: {data_time / (i+1)}"
                )
                wandb_run.log({
                    "step": step,
                    "lr": scheduler_0.get_last_lr()[0],
                    "forward_loss" if self.mode == "direct" else "EIt_loss": mse_0,
                    "backward_loss" if self.mode == "direct" else "Ez_loss": mse_1,
                    "forward_gradnorm" if self.mode == "direct" else "EIt_gradnorm": norm_0,
                    "backward_gradnorm" if self.mode == "direct" else "Ez_gradnorm": norm_1,
                    "epoch": epoch
                })

                data_start = time()

            if self.args.save_every is not None and epoch % self.args.save_every == 0 and epoch > 1:
                self.logger.info(f"Evaluating...")

                state_dict = {
                    "forward" if self.mode == "direct" else "EIt": model_0.state_dict(),
                    "backward" if self.mode == "direct" else "Ez": model_1.state_dict(),
                }

                _, _, err_forward, err_backward = self.test(
                    # XXX: fix sampling!
                    state_dict, max_n_samples=None, n_batch_size=self.config["training"]["n_batch"], all_t=False, phase="valid")

                wandb_run.log({
                    "step": step,
                    "forward_valid_rel_l2_err": err_forward,
                    "backward_valid_rel_l2_err": err_backward,
                    "epoch": epoch
                })

                self.logger.info(f"Saving to `{self.args.save_dir}`...")
                os.makedirs(self.args.save_dir, exist_ok=True)
                torch.save(
                    state_dict, f"{self.args.save_dir}/epoch_{epoch}.pth")

    def test(self, state_dict: Mapping[str, Any], max_n_samples: int | None, n_batch_size: int, all_t: bool, phase: Literal["valid", "test"]):
        self.logger.info("testing")

        model_0 = FNO(self.config)
        model_0.load_state_dict(
            state_dict["forward" if self.mode == "direct" else "EIt"])
        model_0 = model_0.to(device=self.device)
        model_0.eval()

        model_1 = FNO(self.config)
        model_1.load_state_dict(
            state_dict["backward" if self.mode == "direct" else "Ez"])
        model_1 = model_1.to(device=self.device)
        model_1.eval()

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
        res_backward = torch.zeros(size=(
            len(times), n_samples, self.dimension) if all_t else (n_samples, self.dimension))

        start_cur = 0
        l2_errs_forward = torch.zeros(
            size=(n_samples, ), dtype=torch.float32, device=self.device)
        l2_errs_backward = torch.zeros(
            size=(n_samples, ), dtype=torch.float32, device=self.device)

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

            if self.mode == "direct":
                X_forward = self.sampler.sample_direct(
                    x0, model_0, self.noise, self.interpolate, times, all_t)
                X_backward = self.sampler.sample_direct(
                    x1, model_1, self.noise, self.interpolate, times, all_t, backward=True)
            elif self.mode == "separate":
                X_forward = self.sampler.sample_separate(
                    x0, model_0, model_1, self.noise, self.interpolate, times, all_t, backward=False)
                X_backward = self.sampler.sample_separate(
                    x1, model_0, model_1, self.noise, self.interpolate, times, all_t, backward=True)
            else:
                raise ValueError()

            if all_t:
                res_forward[:, start_cur:end_cur] = X_forward[:, :, 1]
                res_backward[:, start_cur:end_cur] = X_backward[:, :, 0]
            else:
                res_forward[start_cur:end_cur] = X_forward[:, 1]
                res_backward[start_cur:end_cur] = X_backward[:, 0]

            # now perform evaluation
            if all_t:
                X_forward = X_forward[-1]
                X_backward = X_backward[-1]

            l2_errs_forward[start_cur:end_cur] = (
                X_forward[:, 1] - x1[:, 1]).square().sum(dim=1) / x1[:, 1].square().sum(dim=1)
            l2_errs_backward[start_cur:end_cur] = (
                X_backward[:, 0] - x0[:, 0]).square().sum(dim=1) / x0[:, 0].square().sum(dim=1)

            start_cur = end_cur

        l2_err_forward = float(l2_errs_forward.mean())
        l2_err_backward = float(l2_errs_backward.mean())

        return res_forward, res_backward, l2_err_forward, l2_err_backward

    def test_one(self, state_dict, n_id, n_repeats, all_t):
        raise NotImplementedError()
