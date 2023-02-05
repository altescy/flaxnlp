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
    parser.add_argument("--artifact", type=Path, default=Path("output"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--datamodule", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    config_filename = args.artifact / "config.json"
    checkpoint_filename = args.checkpoint or (args.artifact / "checkpoints")
    datamodule_filename = args.datamodule or (args.artifact / "datamodule.pkl")

    with config_filename.open("r") as jsonfile:
        config = json.load(jsonfile)

    print("Loading datmodule...")
    datamodule = GptDataModule.load(datamodule_filename)

    print("Loading model...")
    model = colt.build(config["model"], colt.Lazy[GPT]).construct(vocab_size=datamodule.vocab_size)
    state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_filename, target=None)

    assert datamodule.token_indexer is not None

    rngs = jax.random.PRNGKey(0)

    with suppress(KeyboardInterrupt, EOFError):
        while True:
            text = input("Enter text: ")
            rngs, subrng = jax.random.split(rngs)
            token_ids = datamodule.token_indexer(datamodule.tokenizer(text))["token_ids"][None, :]
            token_ids = model.generate(
                rngs=subrng,
                params=state["params"],
                token_ids=token_ids,
                max_new_tokens=args.max_new_tokens,
                topk=args.topk,
                temperature=args.temperature,
            )[0]
            tokens = [datamodule.token_indexer.get_value_by_index(index) for index in token_ids]
            if "<eos>" in tokens:
                tokens = tokens[: tokens.index("<eos>")]
            print("".join(tokens))


if __name__ == "__main__":
    main()
