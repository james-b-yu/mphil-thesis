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

    @torch.no_grad()
    def sample_known(self, x: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, interp: Interpolate, noise: Noise, times: torch.Tensor):
        assert times.dim() == 1 and len(times) > 1
        raise NotImplementedError()

    @torch.no_grad()
    def sample_direct(self, start: torch.Tensor, model: nn.Module, noise: Noise, interp: Interpolate, times: torch.Tensor, all_t: bool, backward: bool = False):
        """Euler-Maruyama sampling

        Args:
            model (nn.Module): _description_
        """

        # model provides the drift coefficients
        assert times.dim() == 1 and len(times) > 1

        x = start

        dt = 0

        if all_t:
            res = torch.zeros(size=(len(times), *x.shape), device=x.device)

        for i, t in enumerate(tqdm(times, leave=False)):
            if i + 1 < len(times):
                dt = (times[i + 1] - times[i]).item()  # allow for dynamic dt

            t = t.item()

            t_input = torch.full((x.shape[0], ), fill_value=t, device=x.device)

            z = noise.sample(x.shape[:2])

            drift = model(x, t_input) if not backward else model(
                x, 1.0 - t_input)

            diffusion = z * \
                math.sqrt(2.0 * interp.eps * dt)

            x = x + drift * dt + diffusion

            if all_t:
                res[i] = x

        return res if all_t else x

    @torch.no_grad()
    def sample_separate(self, start: torch.Tensor, model_EIt: nn.Module, model_Ez: nn.Module, noise: Noise, interp: Interpolate, times: torch.Tensor, all_t: bool, backward: bool = False):
        """Euler-Maruyama sampling

        Args:
            model (nn.Module): _description_
        """

        # model provides the drift coefficients
        assert times.dim() == 1 and len(times) > 1

        x = start

        dtheta = 0

        if all_t:
            res = torch.zeros(size=(len(times), *x.shape), device=x.device)

        for i, s in enumerate(tqdm(times, leave=False)):
            theta = interp.theta(s.item()).item()
            theta_input = torch.full(
                (x.shape[0], ), fill_value=theta, device=x.device)

            z = noise.sample(x.shape[:2])

            if i + 1 < len(times):
                dtheta = (times[i + 1] - times[i]).item() * \
                    interp.theta_t(s).item()  # allow for dynamic dt

            if not backward:
                drift = model_EIt(x, theta_input) + interp.get_weight_on_Ez(
                    theta_input, False)[:, None, None] * model_Ez(x, theta_input)
            else:
                drift = -model_EIt(x, 1.0 - theta_input) + interp.get_weight_on_Ez(
                    1.0 - theta_input, True)[:, None, None] * model_Ez(x, 1.0 - theta_input)

            diffusion = z * \
                math.sqrt(2.0 * interp.eps * dtheta)

            x = x + drift * dtheta + diffusion

            if all_t:
                res[i] = x

        return res if all_t else x
