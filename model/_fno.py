from torch import nn
import torch

from ._mlp import MLP, skip_connection, Lifting, Projection
from ._spec_conv import SpectralConv
from config import Config
from torch.nn import functional as F


from ._utils import ddpm_init_, get_t_embedding


def _ddpm_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        ddpm_init_(m.weight, m.bias)


class FNO(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config["fno"] is not None
        self.config = config
        assert self.config["fno"] is not None
        self.order = len(config["fno"]["n_modes"])
        assert self.order == 1, "FNO does not support functions with >1 input dimension"
        self.n_hidden_channels = config["fno"]["n_hidden_channels"]
        self.n_in_channels = config["source_channels"] + \
            config["target_channels"]
        self.n_out_channels = config["source_channels"] + \
            config["target_channels"]
        self.n_lifting_channels = config["fno"]["n_lifting_channels"]
        self.n_layers = config["fno"]["n_layers"]
        self.n_projection_channels = config["fno"]["n_projection_channels"]
        self.skip_type = config["fno"]["skip_type"]

        self.dense = nn.Sequential(
            nn.Linear(config["fno"]["n_lifting_channels"],
                      config["fno"]["n_lifting_channels"]),
            nn.GELU(),  # TODO: i've added this in.
            nn.Linear(config["fno"]["n_lifting_channels"],
                      config["fno"]["n_lifting_channels"])
        )

        self.dense.apply(_ddpm_init)

        self.spec_convs = SpectralConv(config)

        self.skips = nn.ModuleList([skip_connection(self.n_hidden_channels, self.n_hidden_channels,
                                   type=self.skip_type, n_dim=self.order) for _ in range(self.n_layers)])

        self.mlp = nn.ModuleList(
            [MLP(in_channels=self.n_hidden_channels, hidden_channels=int(round(self.n_hidden_channels * config["fno"]["mlp_expansion"])),
                 dropout=config["fno"]["mlp_dropout"], n_dim=self.order, t_embedding_dim=self.n_hidden_channels) for _ in range(self.n_layers)]
        )
        self.mlp_skips = nn.ModuleList([skip_connection(
            self.n_hidden_channels, self.n_hidden_channels, type=self.skip_type, n_dim=self.order) for _ in range(self.n_layers)])

        self.norm = nn.ModuleList([nn.GroupNorm(
            # TODO: test other types of norm
            num_groups=4, num_channels=self.n_hidden_channels) for _ in range(self.n_layers)])

        self.lifting = Lifting(
            in_channels=self.n_in_channels, out_channels=self.n_hidden_channels, n_dim=self.order)
        self.projection = Projection(in_channels=self.n_hidden_channels, out_channels=self.n_out_channels,
                                     # TODO: customize nonlinearity
                                     hidden_channels=self.n_projection_channels, n_dim=self.order)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.lifting(x)

        t_embedding = get_t_embedding(t, self.n_lifting_channels)
        t_embedding = self.dense(t_embedding)

        x = x + t_embedding[:, :, *([None] * self.order)]

        for i in range(self.n_layers):
            x = F.gelu(x)
            x = self.norm[i](x)

            x_conv = self.spec_convs[i](x, t_embedding)
            x_skip = self.skips[i](x)

            x = x_conv + x_skip

            x_skip = self.mlp_skips[i](x)

            if i < (self.n_layers - 1):
                x = F.gelu(x)

            x = self.mlp[i](x) + x_skip

        x = self.projection(x)
        x = x.squeeze(1)
        return x
