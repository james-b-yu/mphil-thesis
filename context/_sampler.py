import math
import torch
from tqdm import tqdm

from ._interp import Interpolate
from ._noise import Noise
from config import Config
from torch import nn


class Sampler:
    def __init__(self, config: Config):
        self.config = config
        self.method = config["sampling"]["method"]

    @torch.no_grad()
    def sample_known(self, x: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, interp: Interpolate, noise: Noise, times: torch.Tensor):
        assert times.dim() == 1 and len(times) > 1
        raise NotImplementedError()

    @torch.no_grad()
    def sample_direct(self, start: torch.Tensor, model: nn.Module, noise: Noise, interp: Interpolate, times: torch.Tensor, all_t: bool):
        """Euler-Maruyama sampling

        Args:
            model (nn.Module): _description_
        """
        raise NotImplementedError()

    @torch.no_grad()
    def sample_separate(self, start: torch.Tensor, model_EIt: nn.Module, model_Ez: nn.Module, noise: Noise, interp: Interpolate, times: torch.Tensor, all_t: bool, backward: bool = False, conditional: bool = False, ode=False, one=False):
        """Euler-Maruyama sampling

        Args:
            model (nn.Module): _description_
        """
        assert not (ode and one)
        if one:
            # one-step ode
            assert not all_t
            theta_input = torch.zeros(
                (start.shape[0], ), device=start.device)

            if not conditional:
                model_input = start
            else:
                cond_start = (start[:, :self.config["source_channels"]
                                    ] if not backward else start[:, self.config["source_channels"]:]) if self.config["layout"] == "product" else start
                model_input = torch.concat((start, cond_start), dim=1)

            if not backward:
                return start + model_EIt(model_input, theta_input)
            else:
                return start - model_EIt(model_input, 1.0 - theta_input)

        else:
            # model provides the drift coefficients
            assert times.dim() == 1 and len(times) > 1

            x = start

            ds = dtheta = 0

            if all_t:
                res = torch.zeros(size=(len(times), *x.shape), device=x.device)

            def get_elements(s: float, xt: torch.Tensor, z: torch.Tensor):
                theta = interp.theta(s).item()
                theta_input = torch.full(
                    (xt.shape[0], ), fill_value=theta, device=x.device).clamp(min=0.0, max=1.0)

                if not conditional:
                    model_input = xt
                else:
                    cond_start = (start[:, :self.config["source_channels"]
                                        ] if not backward else start[:, self.config["source_channels"]:]) if self.config["layout"] == "product" else start
                    model_input = torch.concat((xt, cond_start), dim=1)

                if not backward:
                    drift = model_EIt(model_input, theta_input) + interp.get_weight_on_Ez(
                        theta_input, False, ode)[:, *([None] * (1 + self.config["dimensions"]))] * model_Ez(model_input, theta_input)
                else:
                    drift = -model_EIt(model_input, 1.0 - theta_input) + interp.get_weight_on_Ez(
                        1.0 - theta_input, True, ode)[:, *([None] * (1 + self.config["dimensions"]))] * model_Ez(model_input, 1.0 - theta_input)

                diffusion = z * \
                    math.sqrt(2.0 * interp.eps)

                return drift, diffusion

            for i, s in enumerate(tqdm(times, leave=False)):
                if i + 1 < len(times):
                    # allow for dynamic dt
                    ds = (times[i + 1] - times[i]).item()

                theta_t = interp.theta_t(s).item()

                z = noise.sample(
                    x.shape[:2]) if not ode else torch.zeros_like(x)

                if self.method == "heun" and s + ds < times[-1]:
                    x_orig = x

                    x_hat = x_orig + z * \
                        math.sqrt(2.0 * interp.eps * theta_t * ds)
                    drift, _ = get_elements(s.item() + ds, x_hat, z)
                    theta_t_next = interp.theta_t(s + ds).item()

                    x_p = x_hat + drift * theta_t_next * ds
                    drift_tick, _ = get_elements(s.item() + ds, x_p, z)

                    x = x_hat + 0.5 * (drift + drift_tick) * theta_t_next * ds
                else:
                    drift, diffusion = get_elements(s.item(), x, z)
                    drift, diffusion = drift * \
                        theta_t, diffusion * (theta_t ** 0.5)

                    x_orig = x
                    x = x_orig + drift * ds + diffusion * (ds ** 0.5)

                    if self.method == "em_2" and s + ds < times[-1]:
                        drift_tick, diffusion_tick = get_elements(
                            s.item() + ds, x, z)
                        drift_tick, diffusion_tick = drift_tick * \
                            theta_t, diffusion_tick * (theta_t ** 0.5)
                        x = x_orig + 0.5 * (drift + drift_tick) * ds + \
                            0.5 * (diffusion + diffusion_tick) * (ds ** 0.5)

                    if all_t:
                        res[i] = x

        return res if all_t else x
