#!/usr/bin/env python3
from argparse import Namespace
import logging
from context import HilbertStochasticInterpolant
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

    context = HilbertStochasticInterpolant(args, config, logger)

    if args.command == "train":
        context.train()
    elif args.command == "test":
        state_dict = torch.load(args.pth)
        res_forward, res_backward, err_forward, err_backward = context.test(
            state_dict, args.max_n_samples, args.n_batch_size, args.all_t, phase="test")
        res_forward = res_forward.numpy()
        res_backward = res_backward.numpy()
        print(f"Relative L2 Error. Forward: {err_forward:.2f}. Backward: {err_backward:.2f}")
        print(f"Saving to `{args.out_file}`...")
        np.savez_compressed(args.out_file, forward=res_forward, backward=res_backward)

    elif args.command == "test_one":
        state_dict = torch.load(args.pth)
        res_forward, res_backward = context.test_one(
            state_dict, args.n_id, args.n_repeats, args.all_t)
        np.savez_compressed(
            args.out_file, forward=res_forward.numpy(force=True), backward=res_backward.numpy(force=True))

    elif args.command == "sample":
        raise NotImplementedError()


if __name__ == "__main__":
    main()
