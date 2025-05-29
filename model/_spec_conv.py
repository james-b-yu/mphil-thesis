
import itertools
from typing import cast
from torch import nn
import torch

from config._schema import Config
from ._utils import ddpm_init_, EINSUM_SYMBOLS

from tltorch.factorized_tensors import FactorizedTensor
import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')
use_opt_einsum('optimal')


class Dense(nn.Module):
    def __init__(self, weight: torch.Tensor | FactorizedTensor, n_channels: int, n_t_embedding_dim: int, separable: bool, time_act: nn.Module):
        super().__init__()
        self.weight = weight
        self.separable = separable
        self.time_dense = nn.Linear(n_t_embedding_dim, n_channels)
        self.time_act = time_act
        ddpm_init_(self.time_dense.weight, self.time_dense.bias)
        self.separable = separable

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor):
        x_order = x.dim()

        # batch-size, in_channels, x, y...
        x_symbols = list(EINSUM_SYMBOLS[:x_order])

        # in_channels, x, y, ...
        weight_symbols = x_symbols[1:]

        # batch-size, out_channels, x, y...
        if self.separable:
            out_symbols = [x_symbols[0]] + list(weight_symbols)
        else:
            weight_symbols.insert(1, EINSUM_SYMBOLS[x_order])  # outputs
            out_symbols = list(weight_symbols)
            out_symbols[0] = x_symbols[0]

        eq = ''.join(x_symbols) + ',' + ''.join(weight_symbols) + \
            '->' + ''.join(out_symbols)

        if not torch.is_tensor(self.weight):
            weight = self.weight.to_tensor()

        x += self.time_dense(self.time_act(t_embedding))[:, :, None]

        return tl.einsum(eq, x, weight)


class SpectralConv(nn.Module):
    def __init__(self, config: Config):
        """N-dimensional fourier neural operator

        Args:
            config (Config): 
        """
        super().__init__()

        self.config = config

        # n_modes is total number of modes kept along each dimension
        self.order = len(config["fno"]["n_modes"])
        self.n_modes = config["fno"]["n_modes"]
        self.half_modes = [m // 2 for m in self.n_modes]
        self.n_channels = config["fno"]["n_hidden_channels"]
        self.n_t_embedding_dim = config["fno"]["n_time_embedding_dim"]
        self.scale = 1.0 / (self.n_channels ** 2)
        self.separable = config["fno"]["separable"]
        self.fft_norm = self.config["fno"]["fft_norm"]

        if self.separable:
            self.weight_shape = (self.n_channels, *self.half_modes)
        else:
            self.weight_shape = (
                self.n_channels, self.n_channels, *self.half_modes)

        self.weight = cast(FactorizedTensor, FactorizedTensor.new(
            shape=((2**(self.order-1)), *self.weight_shape),
            factorization=config["fno"]["factorisation_type"],
            rank=config["fno"]["factorisation_rank"],
        ))
        self.weight = self.weight.to(
            device=config["device"])  # need to add this here
        self.weight.normal_(0, self.scale)

        self.bias = nn.Parameter(
            self.scale * torch.randn(*((config["fno"]["n_layers"], self.n_channels) + (1, )*self.order)))
        mode_indexing = [((None, m), (-m, None))
                         for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]
        layers = []

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            if len(self.n_modes) == 1:
                layers.append(Dense(weight=self.weight[i], n_channels=self.n_channels,  # type:ignore
                              n_t_embedding_dim=self.n_t_embedding_dim, separable=self.separable, time_act=nn.SiLU()))
            # TODO: accommodate 2D FNO
            # TODO: allow for sparse implementations
            # TODO: fix if we do not have parameter tying, i.e. separate weights for every layer in  n_layers (need to work with sub_conv)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor, indices: int):
        batch_size, channels, *mode_sizes = x.shape
        fft_sizes = list(mode_sizes)
        fft_sizes[-1] = fft_sizes[-1]//2 + 1  # last coefficient is redundant
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        out_fft = torch.zeros(
            size=[batch_size, channels, *fft_sizes], device=x.device, dtype=torch.cfloat)

        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes were already removed
        mode_indexing = [((None, m), (-m, None))
                         for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b)
                                                      for b in boundaries]

            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            # out_fft[idx_tuple] = self._contract(x[idx_tuple], t, self.weight[indices + i], separable=self.separable)

            out_fft[idx_tuple] = self.layers[i](x[idx_tuple], t_embedding)
        x = torch.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def __getitem__(self, indices):
        return SubConv2d(self, indices)


class SubConv2d(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, t_embedding):
        return self.main_conv.forward(x, t_embedding, self.indices)
