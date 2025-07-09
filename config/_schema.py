from typing import List, Literal, TypedDict
from strictyaml import Map, Enum, Int, Seq, Bool, Float, Str


class Data(TypedDict):
    dataset: Literal["gridwatch", "darcy_1d"]


class Training(TypedDict):
    n_batch:  int
    n_epochs: int
    grad_clip: float
    lr: float
    n_warmup_steps: int
    n_cosine_cycle_steps: int


class Model(TypedDict):
    mode: Literal["direct", "separate"]


class Sampling(TypedDict):
    n_t_steps: int
    start_t: float
    end_t: float
    c: float


class Noise(TypedDict):
    gain: float
    len: float


class Interpolate(TypedDict):
    b: float
    # TODO: ADD OTHER SCHEDULES


class FNO(TypedDict):
    n_in_channels: int
    n_out_channels: int
    n_lifting_channels: int
    n_hidden_channels: int
    n_projection_channels: int
    n_time_embedding_dim: int
    n_modes: List[int]
    n_layers: int
    skip_type: Literal["identity", "linear", "soft-gating"]
    norm_type: Literal["group"]
    factorisation_type: Literal["Dense", "CP", "TT", "Tucker",
                                "ComplexDense", "ComplexCP", "ComplexTT", "ComplexTucker"]
    factorisation_rank: int
    separable: bool
    fft_norm: Literal["forward", "backward", "ortho"]
    mlp_dropout: float
    mlp_expansion: int


class Config(TypedDict):
    device: str
    data: Data
    model: Model
    dimension: int
    interpolate: Interpolate
    training: Training
    sampling: Sampling
    fno: FNO
    noise: Noise


config_schema = Map({
    "device": Str(),
    "data": Map({
        "dataset": Enum(["gridwatch", "darcy_1d", "darcy", "poisson", "helmholtz", "burger", "ns-nonbounded", "ns-bounded"]),
    }),
    "model": Map({
        "mode": Enum(["direct", "separate"]),
    }),
    "dimension": Int(),
    "interpolate": Map({
        "b": Float(),
    }),
    "training": Map({
        "n_batch": Int(),
        "n_epochs": Int(),
        "grad_clip": Float(),
        "lr": Float(),
        "n_warmup_steps": Int(),
        "n_cosine_cycle_steps": Int(),
    }),
    "sampling": Map({
        "n_t_steps": Int(),
        "start_t": Float(),
        "end_t": Float(),
        "c": Float(),
    }),
    "fno": Map({
        "n_in_channels": Int(),
        "n_out_channels": Int(),
        "n_lifting_channels": Int(),
        "n_hidden_channels": Int(),
        "n_projection_channels": Int(),
        "n_time_embedding_dim": Int(),
        "n_modes": Seq(Int()),
        "n_layers": Int(),
        "skip_type": Enum(["identity", "linear", "soft-gating"]),
        "norm_type": Enum(["group"]),
        "factorisation_type": Enum(["Dense", "CP", "TT", "Tucker", "ComplexDense", "ComplexCP", "ComplexTT", "ComplexTucker"]),
        "factorisation_rank": Int(),
        "separable": Bool(),
        "fft_norm": Enum(["forward", "backward", "ortho"]),
        "mlp_dropout": Float(),
        "mlp_expansion": Int(),
    }),
    "noise": Map({
        "gain": Float(),
        "len": Float(),
    }),
})
