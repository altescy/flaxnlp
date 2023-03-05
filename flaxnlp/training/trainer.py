import abc
from logging import getLogger
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Set, Tuple, cast

import colt
import flax
import jax
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
from tqdm.auto import tqdm

from flaxnlp.training.callbacks import Callback
from flaxnlp.training.exceptions import StopEarly

logger = getLogger(__name__)

Array = Any
Model = Any


class TrainState(train_state.TrainState):  # type: ignore[misc,no-untyped-call]
    rngs: Dict[str, Any] = flax.struct.field(default_factory=dict)  # type: ignore[no-untyped-call]
    mutables: Set[str] = flax.struct.field(default_factory=set, pytree_node=False)  # type: ignore[no-untyped-call]
    training_steps: int = 0

    def replicate(self) -> "TrainState":
        return cast("TrainState", flax.jax_utils.replicate(self).replace(rngs={k: shard_prng_key(v) for k, v in self.rngs.items()}))  # type: ignore[no-untyped-call]


class TrainingModule(abc.ABC, colt.Registrable):
    @abc.abstractmethod
    def create_state(
        self,
        rngs: Any,
        trainer: "Trainer",
        model: Model,
        inputs: Dict[str, Any],
    ) -> TrainState:
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(
        self,
        state: TrainState,
        inputs: Dict[str, Any],
    ) -> Tuple[TrainState, Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def eval_step(
        self,
        state: TrainState,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


@TrainingModule.register("default")
class DefaultTrainingModule(TrainingModule):
    def create_state(
        self,
        rngs: Any,
        trainer: "Trainer",
        model: Model,
        inputs: Dict[str, Any],
    ) -> TrainState:
        rngs, init_rngs = model.split_rngs(rngs, additional_keys={"params"}, train=True)
        params = model.init(rngs=init_rngs, train=True, **inputs)
        init_rngs.pop("params")
        state = cast(
            TrainState,
            TrainState.create(  # type: ignore[no-untyped-call]
                apply_fn=model.apply,
                params=params,
                tx=trainer.optimizer,
                rngs=init_rngs,
                mutables=model.mutables,
            ),
        )
        return state

    @jax.jit
    def train_step(
        self,
        state: TrainState,
        inputs: Dict[str, Any],
    ) -> Tuple[TrainState, Array]:
        def loss_fn(variables: Any) -> Tuple[Array, Dict[str, Any]]:
            output, mutables = state.apply_fn(
                variables=variables,
                rngs=state.rngs,
                train=True,
                mutable=list(state.mutables),
                **inputs,
            )
            output["__mutables__"] = mutables
            return output["loss"], output

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, output), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)  # type: ignore[no-untyped-call]

        state = state.replace(
            params=FrozenDict({**state.params, **output.pop("__mutables__")}),  # type: ignore[no-untyped-call]
            training_steps=state.training_steps + 1,
            rngs={k: jax.random.fold_in(v, state.training_steps) for k, v in state.rngs.items()},
        )
        return state, output

    @jax.jit
    def eval_step(
        self,
        state: TrainState,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        output = state.apply_fn(
            variables=state.params,
            rngs=state.rngs,
            train=False,
            **inputs,
        )
        return cast(Dict[str, Any], output)


class Trainer:
    def __init__(
        self,
        train_dataloader: Callable[[Sequence], Iterator[Dict[str, Any]]],
        val_dataloader: Optional[Callable[[Sequence], Iterator[Dict[str, Any]]]] = None,
        optimizer: optax.GradientTransformation = optax.adamw(1e-3),
        max_epochs: int = 10,
        training_module: Optional[TrainingModule] = None,
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.training_module = training_module
        self.callbacks = callbacks or []

    def train(
        self,
        rngs: Any,
        model: Model,
        train_dataset: Sequence,
        val_dataset: Optional[Sequence] = None,
        init_state: Optional[TrainState] = None,
    ) -> TrainState:
        if val_dataset is not None and self.val_dataloader is None:
            raise ValueError("val_dataloader must be provided if val_dataset is provided")

        if self.training_module is None:
            self.training_module = TrainingModule.by_name(model.default_training_module)()

        if init_state is None:
            state = self.training_module.create_state(
                rngs=rngs,
                trainer=self,
                model=model,
                inputs=next(self.train_dataloader(train_dataset)),
            )
        else:
            logger.info("Use given train state")
            state = init_state

        p_train_step = jax.pmap(self.training_module.train_step, "batch", donate_argnums=(0,))
        p_eval_step = jax.pmap(self.training_module.eval_step, "batch")

        state = state.replicate()

        for callback in self.callbacks:
            callback.on_start(self, state)

        try:
            with tqdm(range(1, self.max_epochs + 1), position=0) as epochbar:
                for epoch in epochbar:
                    epochbar.set_description(f"Epoch {epoch}")
                    num_train_batches = 0
                    train_metrics: Dict[str, float] = {}
                    train_dataloader = self.train_dataloader(train_dataset)
                    with tqdm(train_dataloader, desc="Training", position=1, leave=False) as trainbar:
                        for inputs in trainbar:
                            inputs = shard(inputs)  # type: ignore[no-untyped-call]

                            num_train_batches += 1
                            state, output = p_train_step(state, inputs)

                            batch_metrics = {"loss": output["loss"], **output.get("metrics", {})}
                            for key, value in batch_metrics.items():
                                train_metrics[key] = train_metrics.get(key, 0.0) + float(value)

                            for callback in self.callbacks:
                                callback.on_batch(
                                    trainer=self,
                                    train_state=state,
                                    batch_inputs=inputs,
                                    batch_outputs=output,
                                    epoch=epoch,
                                    is_training=True,
                                )

                            trainbar.set_postfix(
                                **{key: f"{value / num_train_batches:.5f}" for key, value in train_metrics.items()}
                            )

                    train_metrics = {key: value / num_train_batches for key, value in train_metrics.items()}

                    val_metrics: Dict[str, float] = {}
                    if val_dataset is not None and self.val_dataloader is not None:
                        num_val_batches = 0
                        val_dataloader = self.val_dataloader(val_dataset)
                        with tqdm(val_dataloader, desc="Validation", position=1, leave=False) as valbar:
                            for inputs in valbar:
                                inputs = shard(inputs)  # type: ignore[no-untyped-call]
                                num_val_batches += 1
                                rngs, val_rngs = model.split_rngs(rngs, train=False)
                                output = p_eval_step(state, inputs)
                                batch_metrics = {"loss": output["loss"], **output.get("metrics", {})}
                                for key, value in batch_metrics.items():
                                    val_metrics[key] = val_metrics.get(key, 0.0) + float(value)

                                for callback in self.callbacks:
                                    callback.on_batch(
                                        trainer=self,
                                        train_state=state,
                                        batch_inputs=inputs,
                                        batch_outputs=output,
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

        if jax.process_index() == 0:
            params = jax.device_get(flax.jax_utils.unreplicate(state.params))  # type: ignore[no-untyped-call]
            state = state.replace(params=params)

        for callback in self.callbacks:
            callback.on_end(self, state)

        return state
