from typing import List, Literal, TypedDict
from strictyaml import Map, Enum, Int, Seq


class Data(TypedDict):
    dataset: Literal["gridwatch"]


class Training(TypedDict):
    n_batch:  int
    n_epochs: int


class FNO(TypedDict):
    n_lifting_channels: int
    n_hidden_channels: int
    n_modes: List[int]
    n_layers: int
    norm_type: Literal["group"]


class Config(TypedDict):
    data: Data
    dimension: int
    training: Training
    fno: FNO


config_schema = Map({
    "data": Map({
        "dataset": Enum(["gridwatch"]),
    }),
    "dimension": Int(),
    "training": Map({
        "n_batch": Int(),
        "n_epochs": Int(),
    }),
    "fno": Map({
        "n_lifting_channels": Int(),
        "n_hidden_channels": Int(),
        "n_modes": Seq(Int()),
        "n_layers": Int(),
        "norm_type": Enum(["group"])
    })
})
