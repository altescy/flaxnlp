import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Iterator

import colt
from collatable.extras.dataset import Dataset
from datautil import GptDataModule

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, max_value: int = 100) -> None:
        self._max_value = max_value

    def __iter__(self) -> Iterator[str]:
        N = self._max_value
        for x, y in itertools.product(range(N), range(N)):
            yield f"{x} + {y} = {x + y}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    with args.config.open() as f:
        config = json.load(f)

    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    datamodule = colt.build(config["datamodule"], GptDataModule)
    datamodule.read_dataset(
        DatasetGenerator(),
        train=True,
        path=args.output / "dataset",
    )

    logger.info("Saving datamodule to %s", args.output)
    datamodule.save(args.output / "datamodule.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
