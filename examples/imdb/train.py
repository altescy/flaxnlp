import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, TypeVar, cast

import collatable
import colt
import datasets
import flax
import jax
import numpy
import optax
from clu import metrics
from collatable import Instance, LabelField, TextField
from collatable.extras.dataloader import DataLoader
from collatable.extras.dataset import Dataset
from datautil import ImdbDataModule
from flax import struct
from flax.training import checkpoints, train_state
from flax.training.common_utils import shard
from model import TextClassifier

from flaxnlp.training.trainer import Trainer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    with open(args.config) as jsonfile:
        config = json.load(jsonfile)

    logger.info("Loading dataset...")
    datamodule = colt.build(config["datamodule"], ImdbDataModule)
    train_dataset = datamodule.read_dataset(datasets.load_dataset("imdb", split="train"), train=True)

    logger.info("Building classifier...")
    classifier = colt.build(config["model"], colt.Lazy[TextClassifier]).construct(
        vocab_size=datamodule.vocab_size,
        num_classes=datamodule.num_classes,
    )

    logger.info("Building trainer...")
    trainer = colt.build(config["trainer"], Trainer)

    logger.info("Start training...")
    state = trainer.train(jax.random.PRNGKey(0), classifier, train_dataset)

    logger.info("Saving model...")
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "config.json", "w") as jsonfile:
        json.dump(config, jsonfile, indent=2, ensure_ascii=False)
    datamodule.save(args.output / "datamodule.pkl")
    checkpoints.save_checkpoint(ckpt_dir=args.output / "checkpoints", target=state, step=0, overwrite=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
