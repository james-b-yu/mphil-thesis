import torch


def kernel_SE(x: torch.Tensor, y: torch.Tensor, gain: float, len: float):
    sq_distances = (x[:, None] - y[None, :]) ** 2

    return gain * torch.exp(-0.5 * sq_distances / (len ** 2))
