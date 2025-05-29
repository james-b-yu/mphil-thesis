from torch import nn
import torch
from torch.nn import functional as F
from ._utils import ddpm_init_


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(f"Got {in_features=} and {out_features=}"
                             "but these two must be the same for soft-gating")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(
            1, self.in_features, *(1,)*n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(
                1, self.in_features, *(1,)*n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations
        """
        if self.bias is not None:
            return self.weight*x + self.bias
        else:
            return self.weight*x


def skip_connection(in_features, out_features, n_dim=2, bias=False, type="soft-gating"):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, optional
        whether to use a bias, by default False
    type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if type.lower() == 'soft-gating':
        return SoftGating(in_features=in_features, out_features=out_features, bias=bias, n_dim=n_dim)
    elif type.lower() == 'linear':
        return getattr(nn, f'Conv{n_dim}d')(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
    elif type.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(
            f"Got skip-connection {type=}, expected one of {'soft-gating', 'linear', 'id'}.")


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 n_layers=2, n_dim=2, dropout=0.,
                 hidden=256, act=nn.SiLU(), t_embedding_dim=None,
                 **kwargs,
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.t_embedding_dim = t_embedding_dim if t_embedding_dim is not None else in_channels
        self.act = act
        self.out_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.dropout = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0. else None

        self.Dense_0 = nn.Linear(self.t_embedding_dim, hidden)
        ddpm_init_(self.Dense_0.weight, self.Dense_0.bias)

        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.fcs.append(
                    Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(
                    Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(
                    Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x, temb=None):
        if temb is not None:
            x = x + self.Dense_0(self.act(temb))[:, :, None]

        for i, fc in enumerate(self.fcs):

            x = fc(x)
            if i < self.n_layers:
                x = F.gelu(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x


class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)


class Projection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_dim: int, hidden_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
