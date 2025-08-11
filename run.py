#!/usr/bin/env python3
from argparse import Namespace
import logging

import pandas as pd
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
        res_forward, res_backward, err_forward, err_backward, mse_forward, mse_backward, s_per_sample = context.test(
            state_dict, args.max_n_samples, args.n_batch_size, args.all_t, phase="test", ode=args.ode, one=args.one, resolutions={
                "target": args.target_resolution,
                "model": args.model_resolution,
            } if args.target_resolution is not None else None)
        res_forward = res_forward.numpy()
        res_backward = res_backward.numpy()
        print(
            f"Relative L2 Error. Forward: {100 * err_forward:.2f} %. Backward: {100 * err_backward:.2f} %. s per sample: {s_per_sample:.2f}s")
        print(
            f"MSE Error. Forward: {mse_forward:.2e}. Backward: {mse_backward:.2e}")
        print(f"Saving to `{args.out_file}`...")
        np.savez_compressed(
            args.out_file, forward=res_forward, backward=res_backward)

        if args.stats_out is not None:
            print(f"Saving stats to `{args.stats_out}`...")
            df = pd.DataFrame([{
                "name": f"{config["data"]["dataset"]}b={config["interpolate"]["b"]}len={config["noise"]['len']}",
                "forward_err": err_forward,
                "inverse_err": err_backward,
                "forward_mse": mse_forward,
                "inverse_mse": mse_backward,
                "s_per_sample": s_per_sample,
            }])

            df.to_csv(args.stats_out, index=False)
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
    elif args.command == "test_one":
        assert config is not None
        context = HilbertStochasticInterpolant(args, config, logger)

        state_dict = torch.load(args.pth)

        res_forward, res_backward, err_forward, err_backward, mse_forward, mse_backward, s_per_sample = context.test_one(
            state_dict, args.n_samples, args.n_id, args.n_batch_size, args.all_t, phase="test", ode=args.ode, one=args.one)

        res_forward = res_forward.numpy()
        res_backward = res_backward.numpy()

        print(f"Results for sample id {args.n_id} over {args.n_samples} runs:")
        print(
            f"Relative L2 Error. Forward: {100 * err_forward:.2f} %. Backward: {100 * err_backward:.2f} %. s per sample: {s_per_sample:.2f}s")
        print(
            f"MSE Error. Forward: {mse_forward:.2e}. Backward: {mse_backward:.2e}")
        print(f"Saving to `{args.out_file}`...")
        np.savez_compressed(
            args.out_file, forward=res_forward, backward=res_backward)

        if args.stats_out is not None:
            print(f"Saving stats to `{args.stats_out}`...")
            df = pd.DataFrame([{
                "name": f"{config['data']['dataset']}b={config['interpolate']['b']}len={config['noise']['len']}",
                "forward_err": err_forward,
                "inverse_err": err_backward,
                "forward_mse": mse_forward,
                "inverse_mse": mse_backward,
                "s_per_sample": s_per_sample,
                "n_id": args.n_id,
                "n_samples": args.n_samples,
            }])

            df.to_csv(args.stats_out, index=False)


if __name__ == "__main__":
    main()
