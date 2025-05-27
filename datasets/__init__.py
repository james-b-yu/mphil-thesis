from typing import Literal
from .gridwatch import GridwatchDataset


def get_dataset(dataset_name: Literal["gridwatch"], *args, **kwargs):
    if dataset_name == "gridwatch":
        return GridwatchDataset(*args, **kwargs)


__all__ = ["get_dataset"]
