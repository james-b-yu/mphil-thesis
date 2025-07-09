#!/usr/bin/env python3
from argparse import Namespace
import logging
from context import HilbertStochasticInterpolant
from my_datasets import generate_darcy_1d, diffusion_pde_preprocess
from config import parse
import numpy as np
import torch


def setup_logging(args: Namespace):
    logging_level = getattr(logging, args.logging_level.upper(), None)
    assert isinstance(logging_level, int)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.setLevel(logging_level)

    return logger


def main():
    args, config = parse()
    logger = setup_logging(args)

    if args.command == "train":
        assert config is not None
        context = HilbertStochasticInterpolant(args, config, logger)

        context.train()
    elif args.command == "test":
        assert config is not None
        context = HilbertStochasticInterpolant(args, config, logger)

        state_dict = torch.load(args.pth)
        res_forward, res_backward, err_forward, err_backward = context.test(
            state_dict, args.max_n_samples, args.n_batch_size, args.all_t, phase="test")
        res_forward = res_forward.numpy()
        res_backward = res_backward.numpy()
        print(
            f"Relative L2 Error. Forward: {100 * err_forward:.2f} %. Backward: {100 * err_backward:.2f} %")
        print(f"Saving to `{args.out_file}`...")
        np.savez_compressed(
            args.out_file, forward=res_forward, backward=res_backward)
    elif args.command == "dataset-gen":
        if args.dataset_name == "darcy_1d":
            generate_darcy_1d(logger, args.dest,
                              args.fineness, args.size, args.seed)
        else:
            raise ValueError()
    elif args.command == "diffusion-pde-preprocess":
        diffusion_pde_preprocess(logger, in_prefix=args.raw_loc,
                                 out_path=args.dest, ds_name=args.dataset_name, seed=args.seed)
    elif args.command == "sample":
        raise NotImplementedError()


if __name__ == "__main__":
    main()
