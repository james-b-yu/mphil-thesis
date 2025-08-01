import torch
from diffusers import UNet2DModel


class UNet2d(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self._model = UNet2DModel(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs).sample
