import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import colt
import datasets
import jax
from collatable.extras.dataloader import DataLoader
from datautil import ImdbDataModule
from flax.training import checkpoints
from flax.training.train_state import TrainState
from model import TextClassifier

logger = logging.getLogger(__name__)

Array = Any


class Accuracy:
    def __init__(self) -> None:
        self.num_correct = 0
        self.num_examples = 0

    def __call__(self, gold: Array, pred: Array) -> None:
        self.num_correct += (gold == pred).sum()
        self.num_examples += gold.shape[0]

    def get_metrics(self) -> Dict[str, float]:
        return {"accuracy": self.num_correct / self.num_examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["train", "test"], default="test")
    parser.add_argument("--config", type=Path, default=Path("output/config.json"))
    parser.add_argument("--datamodule", type=Path, default=Path("output/datamodule.pkl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("output/checkpoints/"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    with open(args.config) as jsonfile:
        config = json.load(jsonfile)

    logger.info("Loading datamodule...")
    datamodule = ImdbDataModule.load(args.datamodule)

    logger.info("Loading model...")
    classifier = colt.build(config["model"], colt.Lazy[TextClassifier]).construct(
        vocab_size=datamodule.vocab_size,
        num_classes=datamodule.num_classes,
    )
    state = checkpoints.restore_checkpoint(ckpt_dir=str(args.checkpoint), target=None)

    logger.info("Loading dataset...")
    test_dataset = datamodule.read_dataset(datasets.load_dataset("imdb", split=args.subset))

    logger.info("Start evaluation...")
    accuracy = Accuracy()
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    for b, batch in enumerate(dataloader, start=1):
        output = classifier.apply(variables=state["params"], train=False, **batch)
        logits = output["logits"]
        pred = logits.argmax(axis=-1)
        gold = batch["label"]
        accuracy(gold, pred)
        print(
            f"\rAccuracy: {accuracy.get_metrics()['accuracy']:.4f} [progress: {100*b/len(dataloader):5.1f}%]",
            end="",
            flush=True,
        )
    print()


if __name__ == "__main__":
    main()
