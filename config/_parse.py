import argparse
import os
from typing import cast
import strictyaml
from ._schema import Config, config_schema


def parse():

    # define the config shared parser
    pipeline_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              add_help=False)
    pipeline_parser.add_argument("--config", type=str, required=True,
                                 help="path to config.yml")
    # define the inference shared parser
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    inference_parser.add_argument(
        "--pth", type=str, required=True, help="path to the .pth model file")
    inference_parser.add_argument("--n-batch-size", type=int, required=False,
                                  default=128, help="how many samples to create at the same time")
    inference_parser.add_argument(
        "--out-file", type=str, required=False, default="./samples.npz", help="output file")
    inference_parser.add_argument(
        "--all-t", type=bool, required=False, default=False, help="whether to preserve all time steps")
    inference_parser.add_argument("--stats-out", type=str, required=False, default=None,
                                  help="if specified, output a csv file to this with all the stats")

    # define the main parser
    parser = argparse.ArgumentParser(
        prog="Hilbert Stochastic Interpolants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True
    )
    parser.add_argument("--logging-level", type=str,
                        choices=["info", "debug", "warning", "critical"], default="info")
    # add subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True)

    # define the train subparser
    train_parser = subparsers.add_parser(
        name="train", help="train the model", parents=[pipeline_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument("--save-every", type=int,
                              required=False, default=500, help="save every x epochs")
    train_parser.add_argument("--save-dir", type=str,
                              required=False, default="./out", help="model save dir")
    train_parser.add_argument(
        "--start-epoch", type=int, required=False, default=1, help="1-indexed")
    train_parser.add_argument("--resume", type=str, required=False,
                              default=None, help="if specified, resume at given .pth file")
    train_parser.add_argument("--n-dataworkers", type=int, default=6)
    train_parser.add_argument("--n-prefetch-factor", type=int, default=None)

    test_parser = subparsers.add_parser(name="test", help="evaluate model using test dataset", parents=[
                                        pipeline_parser, inference_parser])
    test_parser.add_argument(
        "--max-n-samples", type=int, required=False, default=None, help="if specified, limit how many maximum samples to create")

    test_one_parser = subparsers.add_parser(
        name="test_one", help="evaluate one test example", parents=[pipeline_parser, inference_parser])
    test_one_parser.add_argument("--n-repeats", type=int)
    test_one_parser.add_argument("--n-id", type=int)

    diagnose_parser = subparsers.add_parser(name="diagnose", help="", parents=[
                                            pipeline_parser, inference_parser])
    diagnose_parser.add_argument("--n-id", type=int)

    sample_parser = subparsers.add_parser(
        name="sample", help="sample from the model", parents=[pipeline_parser, inference_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sample_parser.add_argument(
        "--n-samples", type=int, required=False, default=5, help="how many samples to create")

    # command for generating my own custom datasets
    dataset_gen_parser = subparsers.add_parser(
        name="dataset-gen", help="generate datasets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dataset_gen_parser.add_argument(
        "dataset_name", type=str, help="name of the dataset", choices=["darcy_1d"])
    dataset_gen_parser.add_argument(
        "--fineness", type=int, default=128, help="how many mesh subdivisions in each dimension"
    )
    dataset_gen_parser.add_argument(
        "--size", type=int, default=10_000, help="dataset size")
    dataset_gen_parser.add_argument(
        "--dest", type=str, default="./data", help="dataset destination")
    dataset_gen_parser.add_argument(
        "--seed", type=int, default=0, help="the seed")

    # command for preprocessing the raw DiffusionPDE datasets
    diffusion_pde_preprocess_parser = subparsers.add_parser(
        name="diffusion-pde-preprocess", help="preprocess raw DiffusionPDE datasets", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    diffusion_pde_preprocess_parser.add_argument("--raw", dest="raw_loc", type=str, default="./data/diffusion_pde_raw",
                                                 help="location of raw folder (should contain 'training' and 'testing' which in turn have subfolders for each dataset. no nested training inside training)")
    diffusion_pde_preprocess_parser.add_argument(
        "--dest", type=str, default="./data", help="dataset destination")
    diffusion_pde_preprocess_parser.add_argument(
        "--seed", type=int, default=0, help="the seed")
    diffusion_pde_preprocess_parser.add_argument(
        "dataset_name", type=str, help="name of the dataset", choices=["all", "darcy", "poisson", "helmholtz", "burger", "ns-nonbounded", "ns-bounded"])

    # parse args and config
    args = parser.parse_args()
    config = None
    if "config" in args:
        if not os.path.exists(args.config):
            raise ValueError("Config not found or invalid")

        with open(args.config, "r") as f:
            config = strictyaml.load(f.read(), schema=config_schema)

        assert isinstance(config, strictyaml.YAML), "Invalid YAML"

        config = cast(Config, config.data)

    return args, config
