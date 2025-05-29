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
    elif args.command == "sample":
        state_dict = torch.load(args.pth)
        res = context.sample(state_dict, args.n_samples,
                             args.n_batch_size, args.all_t)
        res = res.numpy()
        np.savez_compressed(args.out_file, res)


if __name__ == "__main__":
    main()
