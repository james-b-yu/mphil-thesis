from typing import List, Literal, TypedDict
from strictyaml import Map, Enum, Int, Seq, Bool, Float, Str, Optional, OrValidator, NullNone


class Data(TypedDict):
    dataset: Literal["gridwatch", "darcy_1d"]


class Training(TypedDict):
    n_batch:  int
    n_epochs: int
    grad_clip: float
    lr: float
    n_warmup_steps: int
    n_cosine_cycle_steps: int | None
    n_ema_half_life_steps: int


class Sampling(TypedDict):
    n_t_steps: int
    start_t: float
    end_t: float
    c: float
    use_pc: bool


class Noise(TypedDict):
    gain: float
    len: float


class Interpolate(TypedDict):
    b: float
    schedule: Literal["lerp", "smoothstep"]
    weighting: Literal["none", "exponential-out", "exponential-in",
                       "exponential-in-out", "linear-in", "linear-out", "linear-in-out"]


class FNO(TypedDict):
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


class UNO2d(TypedDict):
    attn_resolutions: list[int]
    num_blocks: int
    fmult: float
    rank: float
    cbase: int
    cres: list[int]
    dropout: float


class UNet2d(TypedDict):
    widths: list[int]
    n_layers_per_block: int
    dropout: float


class Config(TypedDict):
    device: str
    data: Data
    mode: Literal["direct", "separate", "conditional"]
    layout: Literal["same", "product"]
    # e.g., a dataset where functions are evaluated on a 128x128 grid will have dimensions = 2, resolution=128
    dimensions: int
    resolution: int
    # e.g.: if source function has 2-dimensional output, then source_channels = 2
    source_channels: int
    target_channels: int
    interpolate: Interpolate
    training: Training
    sampling: Sampling
    model: Literal["fno", "uno_2d", "unet_2d"]
    fno: FNO | None
    uno_2d: UNO2d | None
    unet_2d: UNet2d | None
    noise: Noise


config_schema = Map({
    "device": Str(),
    "data": Map({
        "dataset": Enum(["gridwatch", "darcy_1d", "darcy", "poisson", "helmholtz", "burger", "ns-nonbounded", "ns-bounded"]),
    }),
    "mode": Enum(["direct", "separate", "conditional"]),
    "layout": Enum(["same", "product"]),
    "dimensions": Int(),
    "resolution": Int(),
    "source_channels": Int(),
    "target_channels": Int(),
    "interpolate": Map({
        "b": Float(),
        "schedule": Enum(["lerp", "smoothstep"]),
        "weighting": Enum(["none", "exponential-out", "exponential-in", "exponential-in-out", "linear-in", "linear-out", "linear-in-out"])
    }),
    "training": Map({
        "n_batch": Int(),
        "n_epochs": Int(),
        "grad_clip": Float(),
        "lr": Float(),
        "n_warmup_steps": Int(),
        "n_cosine_cycle_steps": OrValidator(Int(), NullNone()),
        "n_ema_half_life_steps": Int(),
    }),
    "sampling": Map({
        "n_t_steps": Int(),
        "start_t": Float(),
        "end_t": Float(),
        "c": Float(),
        "use_pc": Bool(),
    }),
    "model": Enum(["fno", "uno_2d", "unet_2d"]),
    Optional("fno", default=None): Map({
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
    Optional("uno_2d", default=None): Map({
        "attn_resolutions": Seq(Int()),
        "num_blocks": Int(),
        "fmult": Float(),
        "rank": Float(),
        "cbase": Int(),
        "cres": Seq(Int()),
        "dropout": Float(),
    }),
    Optional("unet_2d", default=None): Map({
        "widths": Seq(Int()),
        "n_layers_per_block": Int(),
        "dropout": Float(),
    }),
    "noise": Map({
        "gain": Float(),
        "len": Float(),
    }),
})
