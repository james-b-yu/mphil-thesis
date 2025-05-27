import torch
import numpy as np
from torch import nn


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(
            shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device: str | torch.device = device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError(
                "invalid distribution for variance scaling initializer")

    return init


def ddpm_init_(weight: torch.Tensor, bias: torch.Tensor, scale=1.0, ):
    """initialise same as ddpm

    Args:
        scale (float, optional): _description_. Defaults to 1.0.
    """

    scale = 1e-10 if scale == 0.0 else scale
    vs = variance_scaling(scale, 'fan_avg', 'uniform')

    weight.data = vs(weight.data.shape, weight.data.dtype, weight.data.device)
    nn.init.zeros_(bias)
