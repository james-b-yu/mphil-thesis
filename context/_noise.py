import torch

from ._kernels import kernel_SE
from config._schema import Config


class Noise:
    def __init__(self, config: Config):
        self.config = config
        self.device = config["device"]
        self.n_mesh = config["dimension"]
        # TODO: allow 2D noise
        x = torch.linspace(0, 1, self.n_mesh, device=self.device)

        # gram matrix K
        K = kernel_SE(x, x, config["noise"]
                      ["gain"], config["noise"]["len"])

        L, _ = torch.linalg.cholesky_ex(
            K + 1e-5 * torch.eye(self.n_mesh, device=K.device))

        assert torch.is_tensor(L)

        self.L = L

    def sample(self, size):
        assert size[-1] == self.n_mesh

        z = torch.randn(size=size, device=self.device)
        return z @ self.L.T
