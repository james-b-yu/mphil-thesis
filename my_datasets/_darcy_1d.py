# XXX: compare this file

# dataset generation taken from https://github.com/tyler-ingebrand/OperatorFunctionEncoder/blob/main/src/Datasets/DarcyDataset.py

from concurrent.futures import ProcessPoolExecutor
from functools import cache
import itertools
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import multivariate_normal
from tqdm import tqdm, trange
from logging import Logger

from os import path, makedirs
from numpy.random import default_rng, Generator


def _K_chol(x, l: float, sigma: float):
    x = np.array(x)
    K = sigma * np.exp(-0.5 *
                       (x[:, np.newaxis] - x[np.newaxis, :])**2 / l**2)
    K_chol = np.linalg.cholesky(K + np.eye(x.shape[0]) * 1e-8)
    return K_chol


# sample a Gaussian process
L_PERM = 0.04
L_SOLN = 0.25
SIGMA_PERM = 1.0
SIGMA_SOLN = 0.01


def source_function(x, rng: Generator):
    K_chol = _K_chol(tuple(x), L_PERM, SIGMA_PERM)
    ys = K_chol @ rng.standard_normal(size=x.shape[0])

    # ys = multivariate_normal.rvs(mean=np.zeros_like(x), cov=K)
    return ys


def permeability(s):
    return 0.2 + s**2


def solve_fd(n_points, rng: Generator):
    # Finite difference solver
    x = np.linspace(0, 1, n_points)
    dx = x[1] - x[0]
    u = source_function(x, rng)
    s = np.zeros(n_points)

    for _ in range(100):
        kappa = permeability(s)
        main_diag = (kappa[1:] + kappa[:-1])/dx**2
        upper_diag = -kappa[1:-1]/dx**2
        lower_diag = -kappa[1:-1]/dx**2

        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1],  # type:ignore
                  shape=(n_points-2, n_points-2))
        A = csc_matrix(A)
        s_interior = spsolve(A, u[1:-1])
        s[1:-1] = s_interior  # type:ignore

    return x, s, u


def generate_one(idx: int, seed, grid_fineness: int):
    # generate darcy solution
    rng = default_rng(seed + idx)

    x_fd, s_fd, u_fd = solve_fd(grid_fineness, rng)

    # pull out the operator terms
    x = x_fd
    f_x = u_fd
    tf_y = s_fd

    return x, f_x, tf_y


def generate_dataset(logger: Logger, dest_dir: str, grid_fineness=128, n_functions=1000, seed=0):
    dest_dir = path.realpath(dest_dir)
    if not path.isdir(dest_dir):
        raise ValueError("destination directory must exist")
    dest_dir = path.join(dest_dir, "darcy_1d")
    makedirs(dest_dir, exist_ok=True)

    # Solve and compare both methods
    np.random.seed(0)

    fig, axs = plt.subplots(4, 6, figsize=(15, 10), dpi=300)
    for i in range(24):
        x_fd, s_fd, u_fd = solve_fd(grid_fineness, default_rng(seed=seed + i))
        # print(x_fd.shape, x_fem.shape)

        ax = axs[i//6, i % 6]
        ax.plot(x_fd, s_fd, 'b', linewidth=3, label='FD Solution')
        ax.plot(x_fd, u_fd, 'g', linewidth=3, label='Source Term')

    plt.legend()
    plt.savefig(path.join(dest_dir, "visualisation.pdf"),
                format='pdf', dpi=300, bbox_inches='tight')

    # generate the dataset
    xs, f_xs, tf_ys = [], [], []
    with ProcessPoolExecutor() as executor:
        for x, f_x, tf_y in tqdm(executor.map(generate_one, range(n_functions), itertools.repeat(seed), itertools.repeat(grid_fineness)), total=n_functions, desc="Generating 1D Darcy Dataset"):
            # save
            xs.append(x)
            f_xs.append(f_x)
            tf_ys.append(tf_y)

    # convert to tensor
    xs = torch.tensor(np.array(xs))
    f_xs = torch.tensor(np.array(f_xs))
    tf_ys = torch.tensor(np.array(tf_ys))

    # train/test split
    n_train = int(0.8 * n_functions)
    n_valid = int(0.1 * n_functions)
    train_xs, train_f_xs, train_tf_ys = xs[:
                                           n_train], f_xs[:n_train], tf_ys[:n_train]
    valid_xs, valid_f_xs, valid_tf_ys = xs[n_train: n_train + n_valid], f_xs[n_train: n_train +
                                                                             n_valid], tf_ys[n_train:n_train + n_valid]
    test_xs, test_f_xs, test_tf_ys = xs[n_train + n_valid:], f_xs[n_train +
                                                                  n_valid:], tf_ys[n_train + n_valid:]

    # save
    train = {"x": train_xs, "f": train_f_xs,
             "tf_y": train_tf_ys}
    valid = {"x": valid_xs, "f": valid_f_xs,
             "tf_y": valid_tf_ys}
    test = {"x": test_xs, "f": test_f_xs, "tf_y": test_tf_ys}
    torch.save(train, path.join(dest_dir, "train.pt"))
    torch.save(valid, path.join(dest_dir, "valid.pt"))
    torch.save(test, path.join(dest_dir, "test.pt"))

    created_files = [
        path.join(dest_dir, "train.pt"),
        path.join(dest_dir, "valid.pt"),
        path.join(dest_dir, "test.pt"),
        path.join(dest_dir, "visualisation.pdf"),
    ]

    logger.info(f"Created\n{"\n".join(created_files)}")


class Darcy1dDataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(self, loc: str, phase: Literal["train", "valid", "test"], seed: int = 0, target_resolution: int = 128, layout="product"):
        super().__init__()

        assert target_resolution == 128
        assert layout == "product"

        loc = path.realpath(loc)
        if not path.isdir(loc):
            raise ValueError(f"path {loc} does not exist")

        self.phase = phase
        raw = torch.load(path.join(loc, f"{phase}.pt"))

        self.sensor_locs: torch.Tensor = raw["x"]
        self.permeability: torch.Tensor = raw["f"]
        self.solutions: torch.Tensor = raw["tf_y"]

        self.rng = default_rng()
        self.resolution: int = raw["x"].shape[-1]

        x = self.sensor_locs[0]  # XXX: allow for variable sensor locs
        self.K_chol_perm = _K_chol(x, L_PERM, SIGMA_PERM)
        self.K_chol_soln = _K_chol(x, L_SOLN, SIGMA_SOLN)

        self.means = torch.as_tensor(
            [self.permeability.mean(), self.solutions.mean()], dtype=torch.float32)
        self.stds = torch.as_tensor(
            [self.permeability.std(), self.solutions.std()], dtype=torch.float32)

    @torch.no_grad()
    def __getitem__(self, idx):
        # XXX: fix this bit in the final code
        # perm_prior = (
        #     self.K_chol_perm @ self.rng.standard_normal(size=self.resolution))
        # soln_prior = self.K_chol_soln @ self.rng.standard_normal(
        #     size=self.resolution)

        perm = self.permeability[idx]
        soln = self.solutions[idx]

        perm = (perm - self.means[0]) / self.stds[0]
        soln = (soln - self.means[1]) / self.stds[1]

        perm_prior = torch.zeros_like(perm)
        soln_prior = torch.zeros_like(soln)

        x0 = torch.stack((perm, soln_prior))
        x1 = torch.stack((perm_prior, soln))

        return x0, x1

    def __len__(self):
        return self.sensor_locs.shape[0]
