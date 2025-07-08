from typing import Literal
from .gridwatch import GridwatchDataset
from ._darcy_1d import Darcy1dDataset, generate_dataset as generate_darcy_1d


def get_dataset(dataset_name: Literal["gridwatch", "darcy_1d"], *args, **kwargs):
    if dataset_name == "gridwatch":
        return GridwatchDataset(*args, **kwargs)
    elif dataset_name == "darcy_1d":
        return Darcy1dDataset(loc="./data/darcy_1d", *args, **kwargs)
    raise ValueError()


__all__ = ["get_dataset", "generate_darcy_1d"]
