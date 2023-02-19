from logging import getLogger
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, cast

import jax
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from tqdm.auto import tqdm

from flaxnlp.models.model import Model
from flaxnlp.training.callbacks import Callback
from flaxnlp.training.exceptions import StopEarly

logger = getLogger(__name__)

Array = Any


class TrainState(train_state.TrainState):  # type: ignore[misc]
    training_steps: int = 0


class Trainer:
    def __init__(
        self,
        train_dataloader: Callable[[Sequence], Iterator[Dict[str, Any]]],
        val_dataloader: Optional[Callable[[Sequence], Iterator[Dict[str, Any]]]] = None,
        optimizer: optax.GradientTransformation = optax.adamw(1e-3),
        max_epochs: int = 10,
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.callbacks = callbacks or []

    def train(
        self,
        rngs: Any,
        model: Model,
        train_dataset: Sequence,
        val_dataset: Optional[Sequence] = None,
        state: Optional[TrainState] = None,
    ) -> TrainState:
        if val_dataset is not None and self.val_dataloader is None:
            raise ValueError("val_dataloader must be provided if val_dataset is provided")

        @jax.jit
        def train_step(rngs: Any, state: TrainState, inputs: Dict[str, Any]) -> Tuple[TrainState, Array]:
            def loss_fn(variables: Any) -> Tuple[Array, Dict[str, Any]]:
                mutable = list(model.mutables)
                output, mutables = state.apply_fn(
                    variables=variables,
                    rngs=rngs,
                    train=True,
                    mutable=mutable,
                    **inputs,
                )
                output["__mutables__"] = mutables
                return output["loss"], output

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, output), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)  # type: ignore[no-untyped-call]
            state = state.replace(
                params=FrozenDict({**state.params, **output.pop("__mutables__")}),
                training_steps=state.training_steps + 1,
            )
            return state, output

        @jax.jit
        def val_step(state: TrainState, inputs: Dict[str, Any]) -> Dict[str, Any]:
            output = state.apply_fn(
                variables=state.params,
                rngs=rngs,
                train=False,
                **inputs,
            )
            return cast(Dict[str, Any], output)

        if state is None:
            rngs, init_rngs = model.split_rngs(rngs, additional_keys={"params"}, train=True)
            params = model.init(
                rngs=init_rngs,
                train=True,
                **next(self.train_dataloader(train_dataset)),
            )
            state = TrainState.create(  # type: ignore[no-untyped-call]
                apply_fn=model.apply,
                params=params,
                tx=self.optimizer,
            )
        else:
            logger.info("Use given train state")

        assert state is not None

        for callback in self.callbacks:
            callback.on_start(self, state)

        training_step = 0
        try:
            with tqdm(range(1, self.max_epochs + 1), position=0) as epochbar:
                for epoch in epochbar:
                    epochbar.set_description(f"Epoch {epoch}")
                    num_train_batches = 0
                    train_metrics: Dict[str, float] = {}
                    train_dataloader = self.train_dataloader(train_dataset)
                    with tqdm(train_dataloader, desc="Training", position=1, leave=False) as trainbar:
                        for inputs in trainbar:
                            num_train_batches += 1
                            rngs, train_rngs = model.split_rngs(rngs, train=True)
                            state, output = train_step(train_rngs, state, inputs)

                            batch_metrics = {"loss": output["loss"], **output.get("metrics", {})}
                            for key, value in batch_metrics.items():
                                train_metrics[key] = train_metrics.get(key, 0) + value

                            for callback in self.callbacks:
                                callback.on_batch(
                                    trainer=self,
                                    train_state=state,
                                    batch_inputs=inputs,
                                    batch_outputs=output,
                                    epoch=epoch,
                                    training_step=training_step,
                                    is_training=True,
                                )

                            trainbar.set_postfix(
                                **{key: f"{value / num_train_batches:.5f}" for key, value in train_metrics.items()}
                            )

                    train_metrics = {key: value / num_train_batches for key, value in train_metrics.items()}

                    assert state is not None

                    val_metrics: Dict[str, float] = {}
                    if val_dataset is not None and self.val_dataloader is not None:
                        num_val_batches = 0
                        val_dataloader = self.val_dataloader(val_dataset)
                        with tqdm(val_dataloader, desc="Validation", position=1, leave=False) as valbar:
                            for batch in valbar:
                                num_val_batches += 1
                                rngs, val_rngs = model.split_rngs(rngs, train=False)
                                output = val_step(state, batch)
                                batch_metrics = {"loss": output["loss"], **output.get("metrics", {})}
                                for key, value in batch_metrics.items():
                                    val_metrics[key] = val_metrics.get(key, 0) + value

                                for callback in self.callbacks:
                                    callback.on_batch(
                                        trainer=self,
                                        train_state=state,
                                        batch_inputs=batch,
                                        batch_outputs=output,
                                        epoch=epoch,
                                        training_step=training_step,
                                        is_training=False,
                                    )

                                valbar.set_postfix(
                                    **{key: f"{value / num_val_batches:.5f}" for key, value in val_metrics.items()}
                                )

                        val_metrics = {key: value / num_val_batches for key, value in val_metrics.items()}

                    metrics: Dict[str, float] = {}
                    metrics.update({f"train_{key}": value for key, value in train_metrics.items()})
                    metrics.update({f"val_{key}": value for key, value in val_metrics.items()})

                    for callback in self.callbacks:
                        callback.on_epoch(self, state, epoch, metrics)
        except StopEarly:
            logger.info("Stopping early!")

        for callback in self.callbacks:
            callback.on_end(self, state)

        return cast(TrainState, state)
