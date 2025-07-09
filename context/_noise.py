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
        assert isinstance(self.dimensions, list)

        self.total_points = int(np.prod(self.dimensions))

        Ks = []

        # create a 1d kernel matrix for each input dimension; dim is the number of grid points in the direction of that dimension
        for dim in self.dimensions:
            x = np.linspace(0, 1, dim)
            K = config["noise"]["gain"] * np.exp(-0.5 *
                                                 (x[:, None] - x[None, :])**2 / config["noise"]["len"]**2)
            Ks.append(K)

        # combine 1d kernels for each dimension into an overall K-matrix via Kronecker product
        K = reduce(np.kron, Ks)

        # gram matrix K
        K_chol = np.linalg.cholesky(K + np.eye(self.total_points) * 1e-8)

        self.L = torch.from_numpy(K_chol).to(
            device=config["device"], dtype=torch.float32)

    def sample(self, batch_size: tuple | int = 1):
        if isinstance(batch_size, int):
            batch_size = (batch_size, )

        z = torch.randn(size=(*batch_size, self.total_points),
                        device=self.config["device"], dtype=torch.float32)
        sample_flat = z @ self.L.T
        return sample_flat.view(*batch_size, *self.dimensions)
