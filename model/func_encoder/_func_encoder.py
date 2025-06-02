from typing import Literal, TypedDict
from torch import nn
import torch

from torch.nn import functional as F

from ._mlp_basis import MLPBasis


class FunctionEncoder(nn.Module):
    """learn basis functions over a Hilbert space H
    """

    def __init__(self, domain_shape: torch.Size, range_shape: torch.Size, n_basis: int, hidden_dim: int, non_linearity: str, n_layers: int):
        """initialises a function encoder for functions `f` on a Hilbert space `H`

        `f : X -> Y`

        Args:
            domain_shape: an element `x` in `X` is represented by a tensor of this shape
            range_shape:  an element `y` in `Y` is represented by a tensor of this shape
            n_basis: number of basis functions to use
            hidden_dim: hidden dimension of mlp backbone TODO
            non_linearity:
            n_layers: n layers of mlp backbone
        """
        super().__init__()

        # set hps
        self.domain_shape = domain_shape
        self.range_shape = range_shape
        self.n_basis = n_basis
        self.non_linearity = non_linearity

        assert hasattr(F, non_linearity)

        # register model TODO: allow for other model types
        self.basis = MLPBasis(domain_shape, range_shape,
                              n_basis, hidden_dim, non_linearity, n_layers)

    def to_coefficients(inputs: torch.Tensor, outputs: torch.Tensor)
