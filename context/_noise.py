# XXX: compare this file

import torch

from ._kernels import kernel_SE
from config._schema import Config
import numpy as np


class Noise:
    def __init__(self, config: Config):
        self.config = config
        self.device = config["device"]
        self.n_mesh = config["dimension"]
        # TODO: allow 2D noise
        x = np.linspace(0, 1, self.n_mesh)

        # gram matrix K
        K = config["noise"]["gain"] * np.exp(-0.5 *
                                             (x[:, np.newaxis] - x[np.newaxis, :])**2 / config["noise"]["len"]**2)

        K_chol = np.linalg.cholesky(K + np.eye(x.shape[0]) * 1e-8)

        self.L = torch.from_numpy(K_chol).to(
            device=config["device"], dtype=torch.float32)

    def sample(self, size):
        assert size[-1] == self.n_mesh

        z = torch.randn(size=size, device=self.device)
        return z @ self.L.T
