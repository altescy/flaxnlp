import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

import colt
from datautil import CausalLMDataModule

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, max_value: int = 100) -> None:
        self._max_value = max_value

    def __iter__(self) -> Iterator[str]:
        N = self._max_value
        for x, y in itertools.product(range(N), range(N)):
            yield f"{x} + {y} = {x + y}"


class DatasetReader:
    def __init__(self, filename: Path) -> None:
        self._filename = filename

    def __iter__(self) -> Iterator[str]:
        with open(self._filename, "r") as f:
            for line in f:
                yield line.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--from-file", type=Path, default=None)
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    with args.config.open() as f:
        config = json.load(f)

    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    dataset: Iterable[str]
    if args.from_file is None:
        dataset = DatasetGenerator()
    else:
        dataset = DatasetReader(args.from_file)
    datamodule = colt.build(config["datamodule"], CausalLMDataModule)
    datamodule.read_dataset(dataset, train=True, path=args.output / "dataset")

    logger.info("Saving datamodule to %s", args.output)
    datamodule.save(args.output / "datamodule.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
