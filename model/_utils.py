import torch
import numpy as np
from torch import nn
import math
from torch.nn import functional as F

EINSUM_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


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


def get_t_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    timesteps = 1000*timesteps
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                    device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class EMA:
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module, ema_model: torch.nn.Module, decay: float | None = None, half_life_steps: float | None = None, ):
        if (decay is None and half_life_steps is None) or (decay is not None and half_life_steps is not None):
            raise ValueError(
                "exactly one of 'decay' and 'half_life_steps' must be specified")

        if decay is not None:
            self.decay = decay
            self.one_minus_decay = 1.0 - decay
        else:
            assert half_life_steps is not None
            if half_life_steps <= 0:
                raise ValueError("half-life must be positive")
            elif half_life_steps > 1000:
                self.one_minus_decay = math.log(2) / half_life_steps
                self.decay = 1.0 - self.one_minus_decay
            else:
                self.decay = math.exp(-math.log(2) / half_life_steps)
                self.one_minus_decay = 1.0 - self.decay

        assert 0.0 <= self.decay <= 1.0

        self._model = model  # reference to original model
        self._ema_model = ema_model  # reference to the ema model

    @torch.no_grad()
    def step(self):
        for ema_param, param in zip(self._ema_model.parameters(), self._model.parameters()):
            if param.requires_grad:
                ema_param.copy_(self.decay * ema_param.data +
                                self.one_minus_decay * param)

    def state_dict(self):
        return self.state_dict()
