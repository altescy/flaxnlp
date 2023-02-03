from logging import getLogger
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, cast

import jax
import optax
from flax.training.train_state import TrainState
from tqdm.auto import tqdm

from flaxnlp.models.model import Model
from flaxnlp.training.callbacks import Callback
from flaxnlp.training.exceptions import StopEarly

logger = getLogger(__name__)

Array = Any


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
    ) -> TrainState:
        if val_dataset is not None and self.val_dataloader is None:
            raise ValueError("val_dataloader must be provided if val_dataset is provided")

        @jax.jit
        def train_step(rngs: Any, state: TrainState, inputs: Dict[str, Any]) -> Tuple[TrainState, Array]:
            def loss_fn(variables: Any) -> Array:
                output = state.apply_fn(variables=variables, rngs=rngs, train=True, **inputs)
                return output["loss"]

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)  # type: ignore[no-untyped-call]
            return state, loss

        train_dataloader = self.train_dataloader(train_dataset)
        val_dataloader = (
            self.val_dataloader(val_dataset) if val_dataset is not None and self.val_dataloader is not None else None
        )

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

        for callback in self.callbacks:
            callback.on_start(self, state)

        try:
            with tqdm(range(1, self.max_epochs + 1), position=0) as epochbar:
                for epoch in epochbar:
                    epochbar.set_description(f"Epoch {epoch}")
                    num_train_batches = 0
                    train_metrics: Dict[str, float] = {}
                    with tqdm(train_dataloader, desc="Training", position=1, leave=False) as trainbar:
                        for inputs in trainbar:
                            num_train_batches += 1
                            rngs, train_rngs = model.split_rngs(rngs, train=True)
                            state, loss = train_step(train_rngs, state, inputs)

                            batch_metrics = {"loss": loss}
                            for key, value in batch_metrics.items():
                                train_metrics[key] = train_metrics.get(key, 0) + value

                            for callback in self.callbacks:
                                callback.on_batch(
                                    trainer=self,
                                    train_state=state,
                                    batch_inputs=inputs,
                                    batch_outputs={"loss": loss},
                                    epoch=epoch,
                                    is_training=True,
                                )

                            trainbar.set_postfix(
                                **{key: f"{value / num_train_batches:.5f}" for key, value in train_metrics.items()}
                            )

                    train_metrics = {key: value / num_train_batches for key, value in train_metrics.items()}

                    val_metrics: Dict[str, float] = {}
                    if val_dataloader is not None:
                        num_val_batches = 0
                        with tqdm(val_dataloader, desc="Validation", position=1, leave=False) as valbar:
                            for batch in valbar:
                                num_val_batches += 1
                                rngs, val_rngs = model.split_rngs(rngs, train=False)
                                outputs = state.apply_fn(variabls=state.params, train=False, **batch)
                                batch_metrics = {"loss": outputs["loss"], **outputs.get("metrics", {})}
                                for key, value in batch_metrics.items():
                                    val_metrics[key] = val_metrics.get(key, 0) + value

                                for callback in self.callbacks:
                                    callback.on_batch(
                                        trainer=self,
                                        train_state=state,
                                        batch_inputs=batch,
                                        batch_outputs=outputs,
                                        epoch=epoch,
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
