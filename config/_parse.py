import argparse
import os
from typing import cast
import strictyaml
from ._schema import Config, config_schema


def parse():
    # define the main parser
    parser = argparse.ArgumentParser(
        prog="Hilbert Stochastic Interpolants",
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
    train_parser = subparsers.add_parser(name="train", help="train the model")

    # parse args and config
    args = parser.parse_args()
    if "config" not in args or not os.path.exists(args.config):
        raise ValueError("Config not found or invalid")

    with open(args.config, "r") as f:
        config = strictyaml.load(f.read(), schema=config_schema)

    assert isinstance(config, strictyaml.YAML), "Invalid YAML"

    config = cast(Config, config.data)

    return args, config
