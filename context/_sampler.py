import math
import torch
from tqdm import tqdm

from ._interp import Interpolate
from datasets import get_dataset
from ._noise import Noise
from config import Config
from torch import nn
from torch.utils.data import DataLoader


class Sampler:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, *args, **kwargs):
        return self._sample(*args, **kwargs)

    @torch.no_grad()
    def _sample(self, start: torch.Tensor, model: nn.Module, noise: Noise, times: torch.Tensor, all_t: bool):
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

        for i, t in enumerate(tqdm(times)):
            if i + 1 < len(times):
                dt = (times[i + 1] - times[i]).item()  # allow for dynamic dt

            t = t.item()

            t_input = torch.full((x.shape[0], ), fill_value=t, device=x.device)

            z = noise.sample(x.shape)
            drift = model(x, t_input)

            diffusion = z * \
                math.sqrt(2.0 * self.config["interpolate"]["eps"] * dt)

            x = x + drift * dt + diffusion

            # indices = x.norm(dim=1) > 10
            # x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 17

            if all_t:
                res[i] = x

        return res if all_t else x
