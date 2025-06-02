import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class _FunctionalWrapper(nn.Module):
    """used to wrap nn.F functions into a module
    """

    def __init__(self, fn, *fn_args,  **fn_kwargs):
        super().__init__()
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs

    def forward(self, x):
        return self.fn(x, *self.fn_args, **self.fn_kwargs)


class MLPBasis(nn.Module):
    def __init__(self, domain_shape: torch.Size, range_shape: torch.Size, n_basis: int, hidden_dim: int, non_linearity: str, n_layers: int):
        """
        Args:
            domain_shape: an element `x` in `X` is represented by a tensor of this shape
            range_shape:  an element `y` in `Y` is represented by a tensor of this shape
            n_basis: number of basis functions to use
            non_linearity: which non-linearity to use
        """
        super().__init__()

        self.domain_shape = domain_shape
        self.range_shape = range_shape
        self.n_basis = n_basis
        self.hidden_dim = hidden_dim
        self.non_linearity_f = _FunctionalWrapper(getattr(F, non_linearity))

        self.flattened_domain_dim = int(np.prod(domain_shape))
        self.flattened_range_dim = int(np.prod(range_shape))

        layers = []
        layers.append(nn.Linear(self.flattened_domain_dim, hidden_dim))
        layers.append(self.non_linearity_f)

        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.non_linearity_f)

        layers.append(
            nn.Linear(hidden_dim, n_basis * self.flattened_range_dim))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        assert x.ndim >= len(
            self.domain_shape), f"Expected dimensionality of x to be at least {len(self.domain_shape)}"
        assert tuple(x.shape[-len(self.domain_shape):]
                     ) == tuple(self.domain_shape), f"Expected final dimensions of x to be {self.domain_shape}"

        start_flatten_dim = x.ndim - len(self.domain_shape)

        x = x.flatten(start_flatten_dim)
        x = self.model(x)
        x = x.reshape(tuple([*x.shape[:-1], self.n_basis, *self.range_shape]))

        return x
