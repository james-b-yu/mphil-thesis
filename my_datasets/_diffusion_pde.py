from logging import Logger
from scipy import io
from typing import Literal, TypedDict
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from functools import partial
from time import time
import torch
from torch.utils.data import Dataset as TorchDataset

from tqdm import tqdm

"""
This file aims to perform processing of raw diffusion_pde data files to convert to .arrows
"""


class Info(TypedDict):
    name: str
    mean: float
    std: float


def default_ds_data_process(mat: str, source: Info, target: Info):
    data = io.loadmat(mat)

    n_examples = data[source["name"]].shape[0]

    for n in range(n_examples):
        yield {
            source["name"]: (data[source["name"]][n] - source["mean"]) / source["std"],
            target["name"]: (data[target["name"]][n] -
                             target["mean"]) / target["std"]
        }


def burger_ds_data_process(mat: str, source: Info, target: Info):
    data = io.loadmat(mat)
    input = data["input"]
    output = data["output"][:, 1:, :]

    n_examples = input.shape[0]

    for n in range(n_examples):
        yield {
            source["name"]: input[n],
            target["name"]: output[n],
        }


def ns_nonbounded_ds_data_process(mat: str, source: Info, target: Info):
    data = io.loadmat(mat)
    w0 = data["a"]
    w10 = data["u"][:, :, :, -1]

    n_examples = w0.shape[0]

    for n in range(n_examples):
        yield {
            source["name"]: w0[n],
            target["name"]: w10[n],
        }


def ns_bounded_data_process(npy: str, source: Info, target: Info):
    data = np.load(npy)
    input = data[:, :, :, 4]
    output = data[:, :, :, 8]

    n_examples = input.shape[0]

    for n in range(n_examples):
        yield {
            source["name"]: (input[n] - source["mean"]) / source["std"],
            target["name"]: (output[n] - target["mean"]) / target["std"]
        }


def default_ds_data_gen(prefix: str, name: str, n_train_files: int, phase: Literal["train", "test"], source: Info, target: Info, process, time: float):
    if phase == "train":
        for i in range(n_train_files):
            candidate_path = Path(prefix).absolute(
            ).resolve().joinpath(f"./training/{name}/{name}_{i + 1}.mat")
            assert candidate_path.is_file(), f"{candidate_path} does not exist"
            print(f"Loading `{candidate_path}`...")
            yield from process(str(candidate_path), source, target)
    elif phase == "test":
        candidate_path = Path(prefix).absolute(
            # add fix for burger
        ).resolve().joinpath(f"./testing/{name if name != "burger" else "burgers"}.mat")
        assert candidate_path.is_file(), f"{candidate_path} does not exist"
        print(f"Loading `{candidate_path}`...")
        yield from process(str(candidate_path), source, target)
    else:
        raise ValueError()


def ns_bounded_data_gen(prefix: str, name: str, n_train_files: int, phase: Literal["train", "test"], source: Info, target: Info, process, time: float):
    if phase == "train":
        for a in ["", "1", "2", "3", "4"]:
            for b in ["0", "1", "2"]:
                candidate_path = Path(prefix).absolute(
                ).resolve().joinpath(f"./training/{name}/{a}{b}/v.npy")
                assert candidate_path.is_file(
                ), f"{candidate_path} does not exist"
                print(f"Loading `{candidate_path}`...")
                yield from process(str(candidate_path), source, target)
    elif phase == "test":
        for a in ["1", "2"]:
            candidate_path = Path(prefix).absolute(
            ).resolve().joinpath(f"./testing/{name}/{a}/v.npy")
            assert candidate_path.is_file(), f"{candidate_path} does not exist"
            print(f"Loading `{candidate_path}`...")
            yield from process(str(candidate_path), source, target)
    else:
        raise ValueError()


# summary statistics are pre-calculated
METADATA = {
    "darcy": {
        "n_train_files": 5,
        "data_process_fn": default_ds_data_process,
        "data_generator": default_ds_data_gen,
        "source": {
            "name": "thresh_a_data",
            "mean": 7.5,
            "std": 4.5,
        },
        "target": {
            "name": "thresh_p_data",
            "mean": 5.69201936e-03,
            "std": 3.79030361e-03,
        }
    },
    "poisson": {
        "n_train_files": 5,
        "data_process_fn": default_ds_data_process,
        "data_generator": default_ds_data_gen,
        "source": {
            "name": "f_data",
            "mean": 0.0,
            "std": 0.29194939640472917,
        },
        "target": {
            "name": "phi_data",
            "mean": 9.672269521807787e-06,
            "std": 0.004174777941808814,
        }
    },
    "helmholtz": {
        "n_train_files": 5,
        "data_process_fn": default_ds_data_process,
        "data_generator": default_ds_data_gen,
        "source": {
            "name": "f_data",
            "mean": -1.2693740798990193e-05,
            "std": 0.28445379748688004,
        },
        "target": {
            "name": "psi_data",
            "mean": 1.0505059464586001e-05,
            "std": 0.004280043882119103,
        }
    },
    "burger": {
        "n_train_files": 5,
        "data_process_fn": burger_ds_data_process,
        "data_generator": default_ds_data_gen,
        "source": {
            "name": "input",
            "mean": 0.0,
            "std": 0.2736656836183428,
        },
        "target": {  # NOTE: output is B by 127 by 128 (second dimension is temporal t = 1, .., 127), as we remove the first temporal step as this is equal to input
            "name": "output",
            "mean": 0.0,
            "std": 0.20020309620292004,
        }
    },
    "ns-nonbounded": {
        "n_train_files": 50,
        "data_process_fn": ns_nonbounded_ds_data_process,
        "data_generator": default_ds_data_gen,
        "source": {
            "name": "a",
            "mean": 0.0,
            "std": 0.2621129,
        },
        "target": {
            "name": "u",
            "mean": 0.0,
            "std": 0.25612658,
        }
    },
    "ns-bounded": {
        "n_train_files": 15,
        "data_process_fn": ns_bounded_data_process,
        "data_generator": ns_bounded_data_gen,
        "source": {
            "name": "input",
            "mean": 1.8076426069737133,
            "std": 1.0097894743540485,
        },
        "target": {
            "name": "output",
            "mean": 2.8761729407711085,
            "std": 1.7223064939240207,
        }
    },
}


def prep_diffusion_pde(logger: Logger, in_prefix: str, out_path: str, ds_name: str, seed=0):
    if ds_name in METADATA:
        logger.info(f"Processing {ds_name}")

        train_all_generator, test_generator = (partial(METADATA[ds_name]["data_generator"],
                                                       prefix=in_prefix,
                                                       name=ds_name,
                                                       n_train_files=METADATA[ds_name]["n_train_files"],
                                                       phase=phase,
                                                       source=METADATA[ds_name]["source"],
                                                       target=METADATA[ds_name]["target"],
                                                       process=METADATA[ds_name]["data_process_fn"],
                                                       ) for phase in ["train", "test"])

        # must used ignored to stop caching
        hf_dataset_train_all = Dataset.from_generator(
            train_all_generator, gen_kwargs={"time": time()})
        hf_dataset_test = Dataset.from_generator(
            test_generator, gen_kwargs={"time": time()})

        assert isinstance(hf_dataset_train_all, Dataset)
        assert isinstance(hf_dataset_test, Dataset)
        # raw data only has a train/test split so we add an additional /valid/ split

        hf_datasets = hf_dataset_train_all.train_test_split(
            test_size=0.1, shuffle=True, seed=seed)
        hf_datasets["valid"] = hf_datasets.pop("test")  # rename test -> valid
        hf_datasets["test"] = hf_dataset_test  # add the actual test dataset

        out = Path(out_path).joinpath(
            f"./{ds_name}").resolve()  # XXX: also split into valid

        logger.info(f"Saving to {out}")
        hf_datasets.save_to_disk(Path(out).resolve())

    elif ds_name == "all":
        for key in tqdm(METADATA):
            prep_diffusion_pde(logger, in_prefix, out_path, key, seed)
    else:
        raise ValueError("dataset does not exist")


class DiffusionPDEDataset(TorchDataset):
    def __init__(self, ds_name: str, loc: str, phase: Literal["train", "valid", "test"], target_resolution: int):
        super().__init__()

        assert ds_name in METADATA

        loc_path = Path(loc).resolve()
        assert loc_path.is_dir()

        self._hf_dataset_dict = load_from_disk(dataset_path=str(loc_path))
        self._hf_dataset_dict.set_format("torch")
        self._source_name = METADATA[ds_name]["source"]["name"]
        self._target_name = METADATA[ds_name]["target"]["name"]

        assert isinstance(self._hf_dataset_dict, DatasetDict)
        assert phase in self._hf_dataset_dict

        raw_resolution = self._hf_dataset_dict[phase][0][self._source_name].shape[-2]
        assert raw_resolution == self._hf_dataset_dict[phase][0][
            self._source_name].shape[-1] == self._hf_dataset_dict[phase][0][self._target_name].shape[-2] == self._hf_dataset_dict[phase][0][self._target_name].shape[-1]

        assert target_resolution < raw_resolution and raw_resolution % target_resolution == 0

        self._downsample_factor = raw_resolution // target_resolution

        self._hf_ds = self._hf_dataset_dict[phase]

    def __len__(self):
        return len(self._hf_ds)

    def __getitem__(self, idx):
        row = self._hf_ds[idx]

        source = torch.as_tensor(row[self._source_name], dtype=torch.float32)[
            ..., ::self._downsample_factor, ::self._downsample_factor]
        target = torch.as_tensor(row[self._target_name], dtype=torch.float32)[
            ..., ::self._downsample_factor, ::self._downsample_factor]

        x0 = torch.stack((source, torch.zeros_like(target)), dim=0)
        x1 = torch.stack((torch.zeros_like(source), target), dim=0)

        return x0, x1
