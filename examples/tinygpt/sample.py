import argparse
import json
from contextlib import suppress
from pathlib import Path

import collatable
import colt
import jax
from datautil import GptDataModule
from flax.training import checkpoints
from model import GPT

from flaxnlp.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("output/checkpoints"))
    parser.add_argument("--datamodule", type=Path, default=Path("output/datamodule.pkl"))
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    with args.config.open("r") as jsonfile:
        config = json.load(jsonfile)

    print("Loading datmodule...")
    datamodule = GptDataModule.load(args.datamodule)

    print("Loading model...")
    model = colt.build(config["model"], colt.Lazy[GPT]).construct(vocab_size=datamodule.vocab_size)
    state = checkpoints.restore_checkpoint(ckpt_dir=args.checkpoint, target=None)

    assert datamodule.token_indexer is not None

    with suppress(KeyboardInterrupt, EOFError):
        while True:
            text = input("Enter text: ")
            token_ids = datamodule.token_indexer(datamodule.tokenizer(text))["token_ids"][None, :]
            token_ids = model.generate(
                rngs=jax.random.PRNGKey(0),
                params=state["params"],
                token_ids=token_ids,
                max_new_tokens=10,
                topk=args.topk,
                temperature=args.temperature,
            )[0]
            tokens = [datamodule.token_indexer.get_value_by_index(index) for index in token_ids]
            if "<eos>" in tokens:
                tokens = tokens[: tokens.index("<eos>")]
            print("".join(tokens))


if __name__ == "__main__":
    main()