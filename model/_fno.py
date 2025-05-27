from torch import nn
from ._spec_conv import SpectralConv
from config import Config


from ._utils import ddpm_init_


def _ddpm_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        ddpm_init_(m.weight, m.bias)


class FNO(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.dense = nn.ModuleList([
            nn.Linear(config["fno"]["n_lifting_channels"],
                      config["fno"]["n_lifting_channels"]),
            nn.Linear(config["fno"]["n_lifting_channels"],
                      config["fno"]["n_lifting_channels"])
        ])

        self.dense.apply(_ddpm_init)

        self.spec_conv = SpectralConv(config)

        self.mlp = nn.ModuleList
