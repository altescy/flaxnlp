import argparse
import json
import logging
from pathlib import Path

import colt
import jax
from collatable.extras.dataset import Dataset
from datautil import GptDataModule
from flax.training import checkpoints
from model import GPT

from flaxnlp.training.trainer import Trainer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--dataset", type=Path, default=Path("output/dataset"))
    parser.add_argument("--datamodule", type=Path, default=Path("output/datamodule.pkl"))
    args = parser.parse_args()

    with args.config.open("r") as jsonfile:
        config = json.load(jsonfile)

    logger.info("Loading dataset...")
    datamodule = GptDataModule.load(args.datamodule)
    dataset = Dataset(args.dataset)

    logger.info("Building model...")
    model = colt.build(config["model"], colt.Lazy[GPT]).construct(vocab_size=datamodule.vocab_size)

    logger.info("Build trainer...")
    trainer = colt.build(config["trainer"], Trainer)

    logger.info("Start training...")
    state = trainer.train(jax.random.PRNGKey(0), model, dataset)

    logger.info("Saving model...")
    with open(args.output / "config.json", "w") as jsonfile:
        json.dump(config, jsonfile, indent=2, ensure_ascii=False)
    checkpoints.save_checkpoint(ckpt_dir=args.output / "checkpoints", target=state, step=0, overwrite=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
