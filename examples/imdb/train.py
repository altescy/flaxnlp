import argparse
import functools
import json
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, cast

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
from collatable.extras.indexer import TokenIndexer
from datautil import ImdbDataModule, WhitespaceTokenizer
from flax import struct
from flax.training import checkpoints, train_state
from flax.training.common_utils import shard
from model import TextClassifier

logger = logging.getLogger(__name__)

T = TypeVar("T")
Array = Any


@struct.dataclass
class Metrics(metrics.Collection):  # type: ignore[misc]
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")  # type: ignore[valid-type]


class Optimizer(optax.GradientTransformation, colt.Registrable):  # type: ignore[misc]
    ...


@Optimizer.register("optax", constructor="from_params")
class OptaxOptimizer(Optimizer):
    @staticmethod
    def from_params(name: str, **kwargs: Any) -> optax.GradientTransformation:
        optimizer = getattr(optax, name)(**kwargs)
        assert isinstance(optimizer, optax.GradientTransformation)
        return optimizer


class TrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    metrics: Metrics


class Trainer:
    def __init__(
        self,
        optimizer: Optimizer,
        batch_size: int,
        num_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(
        self,
        rng: Any,
        model: TextClassifier,
        train_dataset: Dataset[Instance],
    ) -> TrainState:
        @jax.jit
        def train_step(rngs: Any, state: TrainState, batch: Dict[str, Any]) -> Array:
            def loss_fn(variables: Any) -> Array:
                output = state.apply_fn(variables=variables, rngs=rngs, train=True, **batch)
                return output["loss"]

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)  # type: ignore[no-untyped-call]
            return state, loss

        @jax.jit
        def compute_metrics(state: TrainState, batch: Dict[str, Any]) -> TrainState:
            output = state.apply_fn(variables=state.params, train=False, **batch)
            metric_updates = state.metrics.single_from_model_output(
                logits=output["logits"], labels=batch["label"], loss=output["loss"]
            )
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state

        logger.info("Setup training...")
        rng, params_rng, dropout_rng = jax.random.split(rng, 3)
        params = model.init(
            rngs={"params": params_rng, "dropout": dropout_rng},
            train=True,
            **collatable.collate(train_dataset[: self.batch_size]),
        )
        state = TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=params,
            tx=self.optimizer,
            metrics=Metrics.empty(),
        )

        logger.info("Start training...")
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(1, self.num_epochs + 1):
            for b, batch in enumerate(dataloader, start=1):
                rng, dropout_rng = jax.random.split(rng)
                rngs = {"dropout": dropout_rng}
                state, loss = train_step(rngs, state, batch)
                state = compute_metrics(state=state, batch=batch)
                metrics = state.metrics.compute()
                print(
                    f"\rEpoch {epoch}/{self.num_epochs} {100*b/len(dataloader):5.1f}% -"
                    f" Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}",
                    end="",
                    flush=True,
                )

            print()
            state = state.replace(metrics=state.metrics.empty())

        return cast(TrainState, state)


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
    datamodule.save(args.output / "datamodule.pkl")
    checkpoints.save_checkpoint(ckpt_dir=args.output / "checkpoints", target=state, step=0, overwrite=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
