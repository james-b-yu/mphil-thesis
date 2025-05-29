import os
from typing import Any, Mapping
import torch
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
        self.logger.info("training")

        train_dataset = get_dataset(
            self.config["data"]["dataset"], phase="train")
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["training"]["n_batch"], shuffle=True)

        model = FNO(self.config)

        if self.args.resume is not None:
            state_dict = torch.load(self.args.resume)
            model.load_state_dict(state_dict)

        model = model.to(device=self.device)

        optim = AdamW(model.parameters(), amsgrad=True,
                      lr=self.config["training"]["lr"])
        step = 0

        for epoch in range(self.args.start_epoch, self.config["training"]
                           ["n_epochs"] + 1):
            data_start = time()
            data_time = 0

            for i, (_, y) in enumerate(train_loader):
                model.train()

                step += 1
                y = y.to(self.device).squeeze(-1)  # time series
                data_time += time() - data_start

                t = torch.rand(y.shape[0], device=self.device)
                z = self.noise.sample(y.shape)
                # TODO: start with other mu(0) distributions!
                x0 = self.noise.sample(y.shape)

                xt = self.interpolate(t, x0, y, z)
                target_drift = self.interpolate.get_target_drift(t, x0, y, z)

                pred_drift = model(xt, t)

                mse_loss = (target_drift -
                            # TODO: accommodate 2D etc
                            pred_drift).square().sum(dim=1).mean(dim=0)

                optim.zero_grad()
                mse_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config["training"]["grad_clip"]
                )
                optim.step()

                self.logger.info(
                    f"step: {step}, loss: {torch.abs(mse_loss).item()}, data time: {data_time / (i+1)}"
                )

                if self.args.save_every is not None and epoch % self.args.save_every == 0 and epoch > 1:
                    os.makedirs(self.args.save_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               f"{self.args.save_dir}/epoch_{epoch}.pth")

                data_start = time()

    def sample(self, state_dict: Mapping[str, Any], n_samples: int, n_batch_size: int, all_t: bool):
        self.logger.info("sampling")

        model = FNO(self.config)
        model.load_state_dict(state_dict)
        model = model.to(device=self.device)
        model.eval()

        times = torch.linspace(
            self.config["sampling"]["start_t"], self.config["sampling"]["end_t"], self.config["sampling"]["n_t_steps"])

        res = torch.zeros(size=(len(times), n_samples, self.dimension)
                          if all_t else (n_samples, self.dimension))

        start_cur = 0
        while start_cur < n_samples:
            n_batch_size = min(n_batch_size, n_samples - start_cur)

            # TODO: allow for aribtrary start and end
            x = self.noise.sample(size=(n_batch_size, self.dimension))

            x = self.sampler(x, model, self.noise, times, all_t)

            if all_t:
                res[:, start_cur:start_cur + n_batch_size] = x
            else:
                res[start_cur:start_cur + n_batch_size] = x

            start_cur += n_batch_size

        return res
