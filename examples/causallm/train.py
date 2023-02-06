import argparse
import json
import logging
from pathlib import Path

import colt
import jax
from collatable.extras.dataset import Dataset
from datautil import CausalLMDataModule
from flax.training import checkpoints
from model import CausalLM

from flaxnlp.training.trainer import Trainer, TrainState

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--datamodule", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    dataset_filename = args.dataset or (args.output / "dataset")
    datamodule_filename = args.datamodule or (args.output / "datamodule.pkl")
    checkpoint_filename = args.checkpoint or (args.output / "checkpoints")

    dataset_filename.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_filename.parent.mkdir(parents=True, exist_ok=True)

    with args.config.open("r") as jsonfile:
        config = json.load(jsonfile)

    logger.info("Loading dataset...")
    datamodule = CausalLMDataModule.load(datamodule_filename)
    dataset = Dataset(dataset_filename)

    logger.info("Building model...")
    model = colt.build(config["model"], colt.Lazy[CausalLM]).construct(vocab_size=datamodule.vocab_size)

    logger.info("Build trainer...")
    trainer = colt.build(config["trainer"], Trainer)

    logger.info("Start training...")
    state = trainer.train(jax.random.PRNGKey(0), model, dataset)

    logger.info("Saving model...")
    with open(args.output / "config.json", "w") as jsonfile:
        json.dump(config, jsonfile, indent=2, ensure_ascii=False)
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_filename, target=state, step=0, overwrite=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
