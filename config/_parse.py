import argparse
import os
from typing import cast
import strictyaml
from ._schema import Config, config_schema


def parse():
    # define inference variables
    inference_parser = argparse.ArgumentParser(add_help=False)
    inference_parser.add_argument(
        "--pth", type=str, required=True, help="path to the .pth model file")
    inference_parser.add_argument("--n-batch-size", type=int, required=False,
                                  default=128, help="how many samples to create at the same time")
    inference_parser.add_argument(
        "--out-file", type=str, required=False, default="./samples.npz", help="output file")
    inference_parser.add_argument(
        "--all-t", type=bool, required=False, default=False, help="whether to preserve all time steps")

    # define the main parser
    parser = argparse.ArgumentParser(
        prog="Hilbert Stochastic Interpolants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True
    )
    parser.add_argument("--config", type=str, required=True,
                        help="path to config.yml")
    parser.add_argument("--logging-level", type=str,
                        choices=["info", "debug", "warning", "critical"], default="info")
    # add subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True)

    # define the train subparser
    train_parser = subparsers.add_parser(
        name="train", help="train the model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument("--save-every", type=int,
                              required=False, default=500, help="save every x epochs")
    train_parser.add_argument("--save-dir", type=str,
                              required=False, default="./out", help="model save dir")
    train_parser.add_argument(
        "--start-epoch", type=int, required=False, default=1, help="1-indexed")
    train_parser.add_argument("--resume", type=str, required=False,
                              default=None, help="if specified, resume at given .pth file")

    sample_parser = subparsers.add_parser(
        name="sample", help="sample from the model", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[inference_parser])
    sample_parser.add_argument(
        "--n-samples", type=int, required=False, default=5, help="how many samples to create")

    test_parser = subparsers.add_parser(
        name="test", help="evaluate on test set", parents=[inference_parser])
    test_parser.add_argument("--max-n-samples", type=int, default=None)

    test_one_parser = subparsers.add_parser(
        name="test_one", help="evaluate on one test example", parents=[inference_parser])
    test_one_parser.add_argument("--n-repeats", type=int)
    test_one_parser.add_argument("--n-id", type=int)

    # parse args and config
    args = parser.parse_args()
    if "config" not in args or not os.path.exists(args.config):
        raise ValueError("Config not found or invalid")

    with open(args.config, "r") as f:
        config = strictyaml.load(f.read(), schema=config_schema)

    assert isinstance(config, strictyaml.YAML), "Invalid YAML"

    config = cast(Config, config.data)

    return args, config
