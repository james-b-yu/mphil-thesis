# XXX: compare this file

import torch

from ._kernels import kernel_SE
from config._schema import Config
import numpy as np
from functools import reduce


class Noise:
    def __init__(self, config: Config):
        self.config = config
        self.device = config["device"]
        self.dimensions = config["dimensions"]
        self.resolution = config["resolution"]
        self.is_isotropic = config["noise"]["len"] == 0.0

        self.total_points = self.resolution ** self.dimensions

        if not self.is_isotropic:
            # create a 1d kernel matrix for each input dimension; dim is the number of grid points in the direction of that dimension
            x = np.linspace(0, 1, self.resolution)
            K = config["noise"]["gain"] * np.exp(-0.5 *
                                                 (x[:, None] - x[None, :])**2 / config["noise"]["len"]**2)

            # combine 1d kernels for each dimension into an overall K-matrix via Kronecker product
            K = reduce(np.kron, [K] * self.dimensions)

            # gram matrix K
            K_chol = np.linalg.cholesky(K + np.eye(self.total_points) * 1e-8)
            self.K = K

            self.L = torch.from_numpy(K_chol).to(
                device=config["device"], dtype=torch.float32)

    def sample(self, batch_size: tuple | int = 1):
        if isinstance(batch_size, int):
            batch_size = (batch_size, )

        if not self.is_isotropic:
            z = torch.randn(size=(*batch_size, self.total_points),
                            device=self.config["device"], dtype=torch.float32)
            sample_flat = z @ self.L.T
            return sample_flat.view(*batch_size, *([self.resolution] * self.dimensions))
        else:
            z = torch.randn(size=(*batch_size, self.total_points),
                            device=self.config["device"], dtype=torch.float32)
            return z.view(*batch_size, *([self.resolution] * self.dimensions))
