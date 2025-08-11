import os
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict
import torch
from tqdm import tqdm

from ._noise import Noise
from ._interp import Interpolate
from ._sampler import Sampler
from config import Config
from argparse import Namespace
from logging import Logger
from my_datasets import get_dataset
from model import FNO, UNO2d, UNet2d, EMA
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from time import time
from datetime import timedelta
from wandb import init as wandb_init
import math


class TargetResolutionOverride(TypedDict):
    target: int
    model: int


def get_model(config: Config, is_forward: bool, model_img_resolution: int | None = None):
    if config["model"] == "fno":
        return FNO(config, is_forward)
    elif config["model"] == "uno_2d":
        assert config["uno_2d"] is not None

        if config["layout"] == "product":
            n_channels = config["source_channels"] + config["target_channels"]
        else:
            assert config["source_channels"] == config["target_channels"]
            n_channels = config["source_channels"]

        n_extra_in_channels = 0
        if config["mode"] == "conditional":
            n_extra_in_channels = config["source_channels"] if is_forward else config["target_channels"]

        return UNO2d(
            img_resolution=config["resolution"] if model_img_resolution is None else model_img_resolution,
            attn_resolutions=config["uno_2d"]["attn_resolutions"],
            channel_mult=config["uno_2d"]["cres"],
            channel_mult_emb=4,
            channel_mult_noise=1,
            decoder_type="standard",
            disable_skip=False,
            dropout=config["uno_2d"]["dropout"],
            embedding_type="positional",
            encoder_type="standard",
            fmult=config["uno_2d"]["fmult"],
            in_channels=n_channels + n_extra_in_channels,
            out_channels=n_channels,
            label_dim=0,
            label_dropout=0,
            model_channels=config["uno_2d"]["cbase"],
            num_blocks=config["uno_2d"]["num_blocks"],
            rank=config["uno_2d"]["rank"],
            resample_filter=[1, 1],
        )
    elif config["model"] == "unet_2d":
        assert config["unet_2d"] is not None

        if config["layout"] == "product":
            n_channels = config["source_channels"] + config["target_channels"]
        else:
            assert config["source_channels"] == config["target_channels"]
            n_channels = config["source_channels"]

        n_extra_in_channels = 0
        if config["mode"] == "conditional":
            n_extra_in_channels = config["source_channels"] if is_forward else config["target_channels"]

        assert len(config["unet_2d"]["widths"]) == 4

        return UNet2d(
            sample_size=config["resolution"],
            in_channels=n_channels + n_extra_in_channels,
            out_channels=n_channels,
            layers_per_block=config["unet_2d"]["n_layers_per_block"],
            block_out_channels=tuple(
                # type:ignore
                w * config["resolution"] for w in config["unet_2d"]["widths"]),
            dropout=config["unet_2d"]["dropout"],
        )
    else:
        raise ValueError()


class HilbertStochasticInterpolant:
    def __init__(self, args: Namespace, config: Config, logger: Logger):
        self.args = args
        self.config = config
        self.logger = logger
        self.device = config["device"]
        self.noise = Noise(config)
        self.interpolate = Interpolate(config)
        self.device = config["device"]
        self.dimensions = config["dimensions"]
        self.resolution = config["resolution"]
        self.source_channels = config["source_channels"]
        self.target_channels = config["target_channels"]
        self.sampler = Sampler(config)
        self.mode = config["mode"]
        self.n_ema_half_life_steps = config["training"]["n_ema_half_life_steps"]

    def train(self):
        wandb_run = wandb_init(
            name=f"{self.config["data"]["dataset"]};{self.config["mode"]};b={self.config["interpolate"]["b"]};{self.config["interpolate"]["schedule"]};weighted;len={self.config["noise"]['len']}")
        self.logger.info("training")

        train_dataset = get_dataset(
            self.config["data"]["dataset"], phase="train", target_resolution=self.config["resolution"], layout=self.config["layout"])
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["training"]["n_batch"], shuffle=True, num_workers=self.args.n_dataworkers, prefetch_factor=self.args.n_prefetch_factor)

        n_warmup_steps = self.config["training"]["n_warmup_steps"]
        n_cosine_cycle_steps = self.config["training"]["n_cosine_cycle_steps"]
        max_lr = self.config["training"]["lr"]

        resume_steps = self.args.start_epoch * len(train_loader)
        step = 0

        def fractional(x): return x - math.floor(x)

        def lr_function(step_in: int):
            step = step_in + resume_steps
            if step >= n_warmup_steps and n_cosine_cycle_steps is not None:
                return 0.5 * (1 + math.cos(math.pi * fractional((step - n_warmup_steps) / n_cosine_cycle_steps)))
            elif step >= n_warmup_steps:
                return 1.0
            else:
                return (step / n_warmup_steps)

        # forward if direct, EIt if separate
        model_0 = get_model(self.config, is_forward=True,
                            model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
        # backward if direct, Ez if separate
        model_1 = get_model(self.config, is_forward=True,
                            model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
        ema_model_0 = get_model(self.config, is_forward=True,
                                model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
        ema_model_1 = get_model(self.config, is_forward=True,
                                model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
        if self.config["mode"] == "conditional":
            model_2 = get_model(self.config, is_forward=False,
                                model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
            model_3 = get_model(self.config, is_forward=False,
                                model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
            ema_model_2 = get_model(self.config, is_forward=False,
                                    model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)
            ema_model_3 = get_model(self.config, is_forward=False,
                                    model_img_resolution=None if self.args.target_resolution is None else self.args.model_resolution)

        if self.args.resume is not None:
            model_path = Path(self.args.resume).joinpath("./model.pth")
            ema_path = Path(self.args.resume).joinpath("./ema.pth")
            assert model_path.is_file()
            assert ema_path.is_file()

            state_dict = torch.load(model_path)
            ema_state_dict = torch.load(ema_path)

            if self.mode == "direct":
                raise NotImplementedError()
                model_0.load_state_dict(state_dict["forward"])
                model_1.load_state_dict(state_dict["backward"])

                ema_model_0.load_state_dict(ema_state_dict["forward"])
                ema_model_1.load_state_dict(ema_state_dict["backward"])
            elif self.mode == "separate":
                model_0.load_state_dict(state_dict["EIt"])
                model_1.load_state_dict(state_dict["Ez"])

                ema_model_0.load_state_dict(ema_state_dict["EIt"])
                ema_model_1.load_state_dict(ema_state_dict["Ez"])
            elif self.mode == "conditional":
                model_0.load_state_dict(state_dict["EIt_forward"])
                model_1.load_state_dict(state_dict["Ez_forward"])
                model_2.load_state_dict(state_dict["EIt_backward"])
                model_3.load_state_dict(state_dict["Ez_backward"])

                ema_model_0.load_state_dict(ema_state_dict["EIt_forward"])
                ema_model_1.load_state_dict(ema_state_dict["Ez_forward"])
                ema_model_2.load_state_dict(ema_state_dict["EIt_backward"])
                ema_model_3.load_state_dict(ema_state_dict["Ez_backward"])
            else:
                raise ValueError()
        else:
            ema_model_0.load_state_dict(model_0.state_dict())
            ema_model_1.load_state_dict(model_1.state_dict())
            if self.config["mode"] == "conditional":
                ema_model_2.load_state_dict(model_2.state_dict())
                ema_model_3.load_state_dict(model_3.state_dict())

        model_0 = model_0.to(device=self.device)
        model_1 = model_1.to(device=self.device)
        ema_model_0 = ema_model_0.to(device=self.device)
        ema_model_1 = ema_model_1.to(device=self.device)

        ema_update_0 = EMA(model=model_0, ema_model=ema_model_0,
                           half_life_steps=self.n_ema_half_life_steps)
        ema_update_1 = EMA(model=model_1, ema_model=ema_model_1,
                           half_life_steps=self.n_ema_half_life_steps)

        optim_0 = AdamW(model_0.parameters(), amsgrad=True,
                        lr=max_lr)
        optim_1 = AdamW(model_1.parameters(), amsgrad=True,
                        lr=max_lr)

        scheduler_0 = LambdaLR(optim_0, lr_function)
        scheduler_1 = LambdaLR(optim_1, lr_function)

        if self.config["mode"] == "conditional":
            model_2 = model_2.to(device=self.device)
            model_3 = model_3.to(device=self.device)
            ema_model_2 = ema_model_2.to(device=self.device)
            ema_model_3 = ema_model_3.to(device=self.device)

            ema_update_2 = EMA(model=model_2, ema_model=ema_model_2,
                               half_life_steps=self.n_ema_half_life_steps)
            ema_update_3 = EMA(model=model_3, ema_model=ema_model_3,
                               half_life_steps=self.n_ema_half_life_steps)

            optim_2 = AdamW(model_2.parameters(), amsgrad=True, lr=max_lr)
            optim_3 = AdamW(model_3.parameters(), amsgrad=True, lr=max_lr)

            scheduler_2 = LambdaLR(optim_2, lr_function)
            scheduler_3 = LambdaLR(optim_3, lr_function)

        total_steps = (self.config["training"]["n_epochs"] +
                       1 - self.args.start_epoch) * len(train_loader)
        train_start = time()

        self.logger.info(f"Using lr ramp-up of {n_warmup_steps} steps")
        self.logger.info(
            f"Using EMA with a half life of {self.n_ema_half_life_steps} steps, equivalent to a decay rate of {ema_update_0.decay}")
        if n_cosine_cycle_steps is not None:
            self.logger.info(
                f"Cosine annealing with warm restarts of cycle length {n_cosine_cycle_steps} steps")

        for epoch in range(self.args.start_epoch, self.config["training"]
                           ["n_epochs"] + 1):
            data_start = time()
            data_time = 0
            model_0.train()
            model_1.train()

            if self.mode == "conditional":
                model_2.train()
                model_3.train()

            for i, (x0, x1) in enumerate(train_loader):
                step += 1
                x0 = x0.to(self.device, dtype=torch.float32).squeeze(-1)
                x1 = x1.to(self.device, dtype=torch.float32).squeeze(-1)

                data_time += time() - data_start
                elapsed_time = time() - train_start
                avg_time_per_step = elapsed_time / (step)
                eta_time = avg_time_per_step * (total_steps - step)

                t = torch.rand(x1.shape[0], device=self.device)
                z = self.noise.sample(x1.shape[:2])

                if self.mode == "direct":
                    theta = self.interpolate.theta(t)
                    theta_t = self.interpolate.theta_t(t)

                    xt_0 = self.interpolate(theta, x0, x1, z)
                    xt_1 = self.interpolate(1.0 - theta, x0, x1, z)

                    target_0 = self.interpolate.get_target_forward_drift(
                        theta, x0, x1, z) * theta_t[:, *([None] * (1 + self.dimensions))]
                    target_1 = self.interpolate.get_target_backward_drift(
                        theta, x0, x1, z) * theta_t[:, *([None] * (1 + self.dimensions))]
                elif self.mode == "separate":
                    cond_in_0 = cond_in_1 = xt_0 = xt_1 = self.interpolate(
                        t, x0, x1, z)

                    target_0 = self.interpolate.get_target_EIt(t, x0, x1, z)
                    target_1 = self.interpolate.get_target_Ez(t, x0, x1, z)
                elif self.mode == "conditional":
                    xt_0 = xt_1 = xt_2 = xt_3 = self.interpolate(t, x0, x1, z)

                    target_0 = target_2 = self.interpolate.get_target_EIt(
                        t, x0, x1, z)
                    target_1 = target_3 = self.interpolate.get_target_Ez(
                        t, x0, x1, z)

                    cond_in_0 = cond_in_1 = torch.concat(
                        (xt_0, x0[:, :self.source_channels] if self.config["layout"] == "product" else x0), dim=1)
                    cond_in_2 = cond_in_3 = torch.concat(
                        (xt_1, x1[:, self.source_channels:] if self.config["layout"] == "product" else x1), dim=1)
                else:
                    raise ValueError()

                dims = tuple(range(1, 1 + self.dimensions + 1))

                pred_0 = model_0(cond_in_0, t)
                mse_0 = (target_0 -
                         # TODO: accommodate 2D etc
                         pred_0).square().sum(dim=dims).mean(dim=0)
                if torch.isnan(mse_0):
                    self.logger.warning("encountered NAN loss. skipping")
                    continue
                optim_0.zero_grad()
                mse_0.backward()
                norm_0 = torch.nn.utils.clip_grad_norm_(
                    model_0.parameters(
                    ), self.config["training"]["grad_clip"]
                )
                optim_0.step()
                optim_0.zero_grad()

                pred_1 = model_1(cond_in_1, t)
                mse_1 = (target_1 -
                         # TODO: accommodate 2D etc
                         pred_1).square().sum(dim=dims).mean(dim=0)
                if torch.isnan(mse_1):
                    self.logger.warning("encountered NAN loss. skipping")
                    continue
                optim_1.zero_grad()
                mse_1.backward()
                norm_1 = torch.nn.utils.clip_grad_norm_(
                    model_1.parameters(
                    ), self.config["training"]["grad_clip"]
                )
                optim_1.step()
                optim_1.zero_grad()

                ema_update_0.step()
                ema_update_1.step()
                scheduler_0.step()
                scheduler_1.step()

                if self.mode == "conditional":
                    pred_2 = model_2(cond_in_2, t)
                    mse_2 = (target_2 -
                             pred_2).square().sum(dim=dims).mean(dim=0)
                    if torch.isnan(mse_2):
                        self.logger.warning("encountered NAN loss. skipping")
                        continue
                    optim_2.zero_grad()
                    mse_2.backward()
                    norm_2 = torch.nn.utils.clip_grad_norm_(
                        model_2.parameters(
                        ), self.config["training"]["grad_clip"]
                    )
                    optim_2.step()
                    optim_2.zero_grad()

                    pred_3 = model_3(cond_in_3, t)
                    mse_3 = (target_3 -
                             pred_3).square().sum(dim=dims).mean(dim=0)
                    if torch.isnan(mse_3):
                        self.logger.warning("encountered NAN loss. skipping")
                        continue
                    optim_3.zero_grad()
                    mse_3.backward()
                    norm_3 = torch.nn.utils.clip_grad_norm_(
                        model_3.parameters(
                        ), self.config["training"]["grad_clip"]
                    )
                    optim_3.step()
                    optim_3.zero_grad()

                    ema_update_2.step()
                    ema_update_3.step()
                    scheduler_2.step()
                    scheduler_3.step()

                self.logger.info(
                    f"step: {step + resume_steps}, lr: {scheduler_0.get_last_lr()[0]:.2e} epoch: {epoch} ({100 * i / len(train_loader):.1f} %), data time: {data_time / (i+1):.2f}, secs/step: {avg_time_per_step:.2f}, eta: {timedelta(seconds=eta_time)}"
                )

                if self.mode == "direct":
                    wandb_run.log({
                        "step": step + resume_steps,
                        "lr": scheduler_0.get_last_lr()[0],
                        "forward_loss": mse_0,
                        "backward_loss": mse_1,
                        "forward_gradnorm": norm_0,
                        "backward_gradnorm": norm_1,
                        "epoch": epoch,
                        "eta_seconds": eta_time,
                        "seconds_per_step": avg_time_per_step,
                    })
                elif self.mode == "separate":
                    wandb_run.log({
                        "step": step + resume_steps,
                        "lr": scheduler_0.get_last_lr()[0],
                        "EIt_loss": mse_0,
                        "Ez_loss": mse_1,
                        "EIt_gradnorm": norm_0,
                        "Ez_gradnorm": norm_1,
                        "epoch": epoch,
                        "eta_seconds": eta_time,
                        "seconds_per_step": avg_time_per_step,
                    })
                elif self.mode == "conditional":
                    wandb_run.log({
                        "step": step + resume_steps,
                        "lr": scheduler_0.get_last_lr()[0],
                        "EIt_forward_loss": mse_0,
                        "Ez_forward_loss": mse_1,
                        "EIt_forward_gradnorm": norm_0,
                        "Ez_forward_gradnorm": norm_1,
                        "EIt_backward_loss": mse_2,
                        "Ez_backward_loss": mse_3,
                        "EIt_backward_gradnorm": norm_2,
                        "Ez_backward_gradnorm": norm_3,
                        "epoch": epoch,
                        "eta_seconds": eta_time,
                        "seconds_per_step": avg_time_per_step,
                    })
                else:
                    raise ValueError()

                data_start = time()

            if self.args.save_every is not None and epoch % self.args.save_every == 0 and epoch > 1:
                self.logger.info(f"Evaluating...")

                if self.mode == "direct":
                    state_dict = {
                        "forward": model_0.state_dict(),
                        "backward": model_1.state_dict(),
                    }

                    ema_state_dict = {
                        "forward": ema_model_0.state_dict(),
                        "backward": ema_model_1.state_dict(),
                    }
                elif self.mode == "separate":
                    state_dict = {
                        "EIt": model_0.state_dict(),
                        "Ez": model_1.state_dict(),
                    }

                    ema_state_dict = {
                        "EIt": ema_model_0.state_dict(),
                        "Ez": ema_model_1.state_dict(),
                    }
                elif self.mode == "conditional":
                    state_dict = {
                        "EIt_forward": model_0.state_dict(),
                        "Ez_forward": model_1.state_dict(),
                        "EIt_backward": model_2.state_dict(),
                        "Ez_backward": model_3.state_dict(),
                    }

                    ema_state_dict = {
                        "EIt_forward": ema_model_0.state_dict(),
                        "Ez_forward": ema_model_1.state_dict(),
                        "EIt_backward": ema_model_2.state_dict(),
                        "Ez_backward": ema_model_3.state_dict(),
                    }
                else:
                    raise ValueError()

                self.logger.info(f"Saving to `{self.args.save_dir}`...")
                os.makedirs(self.args.save_dir, exist_ok=True)

                torch.save(
                    state_dict, f"{self.args.save_dir}/model.pth")
                torch.save(
                    ema_state_dict, f"{self.args.save_dir}/ema.pth")
                torch.save(
                    state_dict, f"{self.args.save_dir}/epoch_{epoch}.pth")
                torch.save(
                    ema_state_dict, f"{self.args.save_dir}/ema_epoch_{epoch}.pth")

                if self.args.target_resolution is None:
                    _, _, err_forward, err_backward, mse_forward, mse_backward, _ = self.test(
                        state_dict, max_n_samples=256, n_batch_size=self.config["training"]["n_batch"], all_t=False, phase="valid", shuffle=True)
                    _, _, ema_err_forward, ema_err_backward, ema_mse_forward, ema_mse_backward, _ = self.test(
                        ema_state_dict, max_n_samples=256, n_batch_size=self.config["training"]["n_batch"], all_t=False, phase="valid", shuffle=True)

                wandb_run.log({
                    "step": step,
                    "epoch": epoch,
                    "forward_valid_rel_l2_err": err_forward,
                    "backward_valid_rel_l2_err": err_backward,
                    "forward_valid_rel_l2_mse": mse_forward,
                    "backward_valid_rel_l2_mse": mse_backward,
                    "ema_forward_valid_rel_l2_err": ema_err_forward,
                    "ema_backward_valid_rel_l2_err": ema_err_backward,
                    "ema_forward_valid_rel_l2_mse": ema_mse_forward,
                    "ema_backward_valid_rel_l2_mse": ema_mse_backward,
                })

    def test(self, state_dict: Mapping[str, Any], max_n_samples: int | None, n_batch_size: int, all_t: bool, phase: Literal["valid", "test"], shuffle=False, ode=False, one=False, resolutions: TargetResolutionOverride | None = None):
        start_timestamp = time()

        self.logger.info("testing")

        if self.mode != "conditional":
            model_0 = get_model(self.config, is_forward=True,
                                model_img_resolution=None if resolutions is None else resolutions["model"])
            model_0.load_state_dict(
                state_dict["forward" if self.mode == "direct" else "EIt"])
            model_0 = model_0.to(device=self.device)
            model_0.eval()

            model_1 = get_model(self.config, is_forward=True,
                                model_img_resolution=None if resolutions is None else resolutions["model"])
            model_1.load_state_dict(
                state_dict["backward" if self.mode == "direct" else "Ez"])
            model_1 = model_1.to(device=self.device)
            model_1.eval()
        else:
            model_0 = get_model(self.config, is_forward=True,
                                model_img_resolution=None if resolutions is None else resolutions["model"])
            model_0.load_state_dict(
                state_dict["EIt_forward"])
            model_0 = model_0.to(device=self.device)
            model_0.eval()

            model_1 = get_model(self.config, is_forward=True,
                                model_img_resolution=None if resolutions is None else resolutions["model"])
            model_1.load_state_dict(
                state_dict["Ez_forward"])
            model_1 = model_1.to(device=self.device)
            model_1.eval()

            model_2 = get_model(self.config, is_forward=False,
                                model_img_resolution=None if resolutions is None else resolutions["model"])
            model_2.load_state_dict(
                state_dict["EIt_backward"])
            model_2 = model_2.to(device=self.device)
            model_2.eval()

            model_3 = get_model(self.config, is_forward=False,
                                model_img_resolution=None if resolutions is None else resolutions["model"])
            model_3.load_state_dict(
                state_dict["Ez_backward"])
            model_3 = model_3.to(device=self.device)
            model_3.eval()

        times = torch.linspace(
            self.config["sampling"]["start_t"], self.config["sampling"]["end_t"], self.config["sampling"]["n_t_steps"])

        test_dataset = get_dataset(
            self.config["data"]["dataset"], phase=phase, target_resolution=self.config["resolution"], layout=self.config["layout"])
        n_samples = min(max_n_samples, len(test_dataset)
                        ) if max_n_samples is not None else len(test_dataset)
        test_loader = DataLoader(
            test_dataset, batch_size=n_batch_size, shuffle=shuffle)

        resoln_dims = ([self.resolution] * self.dimensions)

        res_forward = torch.zeros(size=(len(times), n_samples, self.target_channels, *resoln_dims)
                                  if all_t else (n_samples, self.target_channels, *resoln_dims))
        res_backward = torch.zeros(size=(
            len(times), n_samples, self.source_channels, *resoln_dims) if all_t else (n_samples, self.source_channels, *resoln_dims))

        start_cur = 0
        l2_errs_forward = torch.zeros(
            size=(n_samples, ), dtype=torch.float32, device=self.device)
        l2_errs_backward = torch.zeros(
            size=(n_samples, ), dtype=torch.float32, device=self.device)
        mse_errs_forward = torch.zeros(
            size=(n_samples, ), dtype=torch.float32, device=self.device)
        mse_errs_backward = torch.zeros(
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
                raise NotImplementedError()
            elif self.mode == "separate" or self.mode == "conditional":
                X_forward = self.sampler.sample_separate(
                    x0, model_0, model_1, self.noise, self.interpolate, times, all_t, backward=False, conditional=self.mode == "conditional", ode=ode, one=one)
                if self.mode == "conditional":
                    X_backward = self.sampler.sample_separate(
                        x1, model_2, model_3, self.noise, self.interpolate, times, all_t, backward=True, conditional=True, ode=ode, one=one)
                else:
                    X_backward = self.sampler.sample_separate(
                        x1, model_0, model_1, self.noise, self.interpolate, times, all_t, backward=True, conditional=False, ode=ode, one=one)
            else:
                raise ValueError()

            if self.config["layout"] == "product":
                if all_t:
                    res_forward[:, start_cur:end_cur] = X_forward[:,
                                                                  :, self.source_channels:]
                    res_backward[:, start_cur:end_cur] = X_backward[:,
                                                                    :, :self.source_channels]
                else:
                    res_forward[start_cur:end_cur] = X_forward[:,
                                                               self.source_channels:]
                    res_backward[start_cur:end_cur] = X_backward[:,
                                                                 :self.source_channels]
            elif self.config["layout"] == "same":
                if all_t:
                    res_forward[:, start_cur:end_cur] = X_forward
                    res_backward[:, start_cur:end_cur] = X_backward
                else:
                    res_forward[start_cur:end_cur] = X_forward
                    res_backward[start_cur:end_cur] = X_backward
            else:
                raise ValueError()

            # now perform evaluation
            if all_t:
                X_forward = X_forward[-1]
                X_backward = X_backward[-1]

            dims = tuple(range(1, 1 + self.dimensions + 1))

            if self.config["layout"] == "product":
                l2_errs_forward[start_cur:end_cur] = (
                    X_forward[:, self.source_channels:] - x1[:, self.source_channels:]).norm(dim=dims, p=2) / x1[:, self.source_channels:].norm(dim=dims, p=2)
                if self.config["data"]["dataset"] == "darcy":
                    l2_errs_backward[start_cur:end_cur] = (~((X_backward[:, :self.source_channels] >= 0) == (
                        x0[:, :self.source_channels] >= 0))).to(dtype=torch.float32).mean(dim=dims)
                else:
                    l2_errs_backward[start_cur:end_cur] = (
                        X_backward[:, :self.source_channels] - x0[:, :self.source_channels]).norm(dim=dims, p=2) / x0[:, :self.source_channels].norm(dim=dims, p=2)

                mse_errs_forward[start_cur:end_cur] = (
                    X_forward[:, self.source_channels:] - x1[:, self.source_channels:]).square().mean(dim=dims)
                mse_errs_backward[start_cur:end_cur] = (
                    X_backward[:, :self.source_channels] - x0[:, :self.source_channels]).square().mean(dim=dims)
            elif self.config["layout"] == "same":
                l2_errs_forward[start_cur:end_cur] = (
                    X_forward - x1).norm(dim=dims, p=2) / x1.norm(dim=dims, p=2)
                l2_errs_backward[start_cur:end_cur] = (
                    X_backward - x0).norm(dim=dims, p=2) / x0.norm(dim=dims, p=2)

                mse_errs_forward[start_cur:end_cur] = (
                    X_forward - x1).square().mean(dim=dims)
                mse_errs_backward[start_cur:end_cur] = (
                    X_backward - x0).square().mean(dim=dims)
            else:
                raise ValueError()

            start_cur = end_cur

        l2_err_forward = float(l2_errs_forward.mean())
        l2_err_backward = float(l2_errs_backward.mean())
        mse_err_forward = float(mse_errs_forward.mean())
        mse_err_backward = float(mse_errs_backward.mean())

        duration_seconds = time() - start_timestamp
        s_per_sample = duration_seconds / (2.0 * n_samples)

        return res_forward, res_backward, l2_err_forward, l2_err_backward, mse_err_forward, mse_err_backward, s_per_sample

    def test_one(self, state_dict: Mapping[str, Any], n_samples: int, n_id: int, n_batch_size: int, all_t: bool, phase: Literal["valid", "test"], ode=False, one=False):
        """
        Tests the model by generating a single specified example multiple times.

        Args:
            state_dict: Dictionary containing the model state dictionaries.
            n_samples: The number of times to generate the single example.
            n_id: The index of the example in the dataset to test.
            n_batch_size: The batch size to use for generation.
            all_t: Whether to return the full time trajectory.
            phase: The dataset phase to use ('valid' or 'test').

        Returns:
            A tuple containing generated samples, error metrics, and performance info.
        """
        start_timestamp = time()
        self.logger.info(f"Testing example {n_id} for {n_samples} samples.")

        if self.mode != "conditional":
            model_0 = get_model(self.config, is_forward=True)
            model_0.load_state_dict(
                state_dict["forward" if self.mode == "direct" else "EIt"])
            model_0 = model_0.to(device=self.device)
            model_0.eval()

            model_1 = get_model(self.config, is_forward=True)
            model_1.load_state_dict(
                state_dict["backward" if self.mode == "direct" else "Ez"])
            model_1 = model_1.to(device=self.device)
            model_1.eval()
        else:
            model_0 = get_model(self.config, is_forward=True)
            model_0.load_state_dict(state_dict["EIt_forward"])
            model_0 = model_0.to(device=self.device)
            model_0.eval()

            model_1 = get_model(self.config, is_forward=True)
            model_1.load_state_dict(state_dict["Ez_forward"])
            model_1 = model_1.to(device=self.device)
            model_1.eval()

            model_2 = get_model(self.config, is_forward=False)
            model_2.load_state_dict(state_dict["EIt_backward"])
            model_2 = model_2.to(device=self.device)
            model_2.eval()

            model_3 = get_model(self.config, is_forward=False)
            model_3.load_state_dict(state_dict["Ez_backward"])
            model_3 = model_3.to(device=self.device)
            model_3.eval()

        times = torch.linspace(
            self.config["sampling"]["start_t"], self.config["sampling"]["end_t"], self.config["sampling"]["n_t_steps"])

        test_dataset = get_dataset(
            self.config["data"]["dataset"], phase=phase, target_resolution=self.config["resolution"], layout=self.config["layout"])

        if n_id >= len(test_dataset):
            raise IndexError(
                f"n_id {n_id} is out of bounds for dataset with length {len(test_dataset)}")

        x0_single, x1_single = test_dataset[n_id]

        x0_single = x0_single.unsqueeze(0).to(
            device=self.device, dtype=torch.float32)
        x1_single = x1_single.unsqueeze(0).to(
            device=self.device, dtype=torch.float32)

        resoln_dims = ([self.resolution] * self.dimensions)
        res_forward = torch.zeros(size=(len(times), n_samples, self.target_channels, *resoln_dims)
                                  if all_t else (n_samples, self.target_channels, *resoln_dims))
        res_backward = torch.zeros(size=(
            len(times), n_samples, self.source_channels, *resoln_dims) if all_t else (n_samples, self.source_channels, *resoln_dims))

        l2_errs_forward = torch.zeros(
            size=(n_samples,), dtype=torch.float32, device=self.device)
        l2_errs_backward = torch.zeros(
            size=(n_samples,), dtype=torch.float32, device=self.device)
        mse_errs_forward = torch.zeros(
            size=(n_samples,), dtype=torch.float32, device=self.device)
        mse_errs_backward = torch.zeros(
            size=(n_samples,), dtype=torch.float32, device=self.device)

        start_cur = 0
        with tqdm(total=n_samples, desc=f"Generating sample {n_id}") as pbar:
            while start_cur < n_samples:
                # Determine the size of the current batch
                current_batch_size = min(n_batch_size, n_samples - start_cur)
                end_cur = start_cur + current_batch_size

                # Create a batch by repeating the single sample
                # This is efficient as it avoids moving the same data to the GPU repeatedly
                x0 = x0_single.expand(
                    current_batch_size, -1, *([-1]*self.dimensions))
                x1 = x1_single.expand(
                    current_batch_size, -1, *([-1]*self.dimensions))

                # --- Sampling ---
                # This logic is identical to the test function
                if self.mode == "direct":
                    X_forward = self.sampler.sample_direct(
                        x0, model_0, self.noise, self.interpolate, times, all_t)
                    X_backward = self.sampler.sample_direct(
                        x1, model_1, self.noise, self.interpolate, times, all_t)
                elif self.mode == "separate" or self.mode == "conditional":
                    X_forward = self.sampler.sample_separate(
                        x0, model_0, model_1, self.noise, self.interpolate, times, all_t, backward=False, conditional=self.mode == "conditional", ode=ode, one=one)
                    if self.mode == "conditional":
                        X_backward = self.sampler.sample_separate(
                            x1, model_2, model_3, self.noise, self.interpolate, times, all_t, backward=True, conditional=True, ode=ode, one=one)
                    else:
                        X_backward = self.sampler.sample_separate(
                            x1, model_0, model_1, self.noise, self.interpolate, times, all_t, backward=True, conditional=False, ode=ode, one=one)
                else:
                    raise ValueError()

                if self.config["layout"] == "product":
                    if all_t:
                        res_forward[:, start_cur:end_cur] = X_forward[:,
                                                                      :, self.source_channels:]
                        res_backward[:, start_cur:end_cur] = X_backward[:,
                                                                        :, :self.source_channels]
                    else:
                        res_forward[start_cur:end_cur] = X_forward[:,
                                                                   self.source_channels:]
                        res_backward[start_cur:end_cur] = X_backward[:,
                                                                     :self.source_channels]
                elif self.config["layout"] == "same":
                    if all_t:
                        res_forward[:, start_cur:end_cur] = X_forward
                        res_backward[:, start_cur:end_cur] = X_backward
                    else:
                        res_forward[start_cur:end_cur] = X_forward
                        res_backward[start_cur:end_cur] = X_backward
                else:
                    raise ValueError()

                X_forward_eval = X_forward[-1] if all_t else X_forward
                X_backward_eval = X_backward[-1] if all_t else X_backward
                dims = tuple(range(1, 1 + self.dimensions + 1))

                if self.config["layout"] == "product":
                    l2_errs_forward[start_cur:end_cur] = (X_forward_eval[:, self.source_channels:] - x1[:, self.source_channels:]).norm(
                        dim=dims, p=2) / x1[:, self.source_channels:].norm(dim=dims, p=2)
                    if self.config["data"]["dataset"] == "darcy":
                        l2_errs_backward[start_cur:end_cur] = (~((X_backward_eval[:, :self.source_channels] >= 0) == (
                            x0[:, :self.source_channels] >= 0))).to(dtype=torch.float32).mean(dim=dims)
                    else:
                        l2_errs_backward[start_cur:end_cur] = (X_backward_eval[:, :self.source_channels] - x0[:, :self.source_channels]).norm(
                            dim=dims, p=2) / x0[:, :self.source_channels].norm(dim=dims, p=2)
                    mse_errs_forward[start_cur:end_cur] = (
                        X_forward_eval[:, self.source_channels:] - x1[:, self.source_channels:]).square().mean(dim=dims)
                    mse_errs_backward[start_cur:end_cur] = (
                        X_backward_eval[:, :self.source_channels] - x0[:, :self.source_channels]).square().mean(dim=dims)
                elif self.config["layout"] == "same":
                    l2_errs_forward[start_cur:end_cur] = (
                        X_forward_eval - x1).norm(dim=dims, p=2) / x1.norm(dim=dims, p=2)
                    l2_errs_backward[start_cur:end_cur] = (
                        X_backward_eval - x0).norm(dim=dims, p=2) / x0.norm(dim=dims, p=2)
                    mse_errs_forward[start_cur:end_cur] = (
                        X_forward_eval - x1).square().mean(dim=dims)
                    mse_errs_backward[start_cur:end_cur] = (
                        X_backward_eval - x0).square().mean(dim=dims)
                else:
                    raise ValueError()

                start_cur = end_cur
                pbar.update(current_batch_size)

        l2_err_forward = float(l2_errs_forward.mean())
        l2_err_backward = float(l2_errs_backward.mean())
        mse_err_forward = float(mse_errs_forward.mean())
        mse_err_backward = float(mse_errs_backward.mean())

        duration_seconds = time() - start_timestamp
        s_per_sample = duration_seconds / (2.0 * n_samples)

        return res_forward, res_backward, l2_err_forward, l2_err_backward, mse_err_forward, mse_err_backward, s_per_sample
