from config import Config
from argparse import Namespace
from logging import Logger
from datasets import get_dataset
from torch.utils.data import DataLoader


class HilbertStochasticInterpolant:
    def __init__(self, args: Namespace, config: Config, logger: Logger):
        self.args = args
        self.config = config
        self.logger = logger

    def train(self):
        self.logger.info("training")

        train_dataset = get_dataset(
            self.config["data"]["dataset"], phase="train")
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["training"]["n_batch"], shuffle=True)

    def sample(self):
        self.logger.info("sampling")
