
from torch import nn

from config._schema import Config

from tltorch.factorized_tensors import FactorizedTensor


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

        self.weight_shape = (self.n_channels, *self.half_modes)

        self.weight = FactorizedTensor.new()
