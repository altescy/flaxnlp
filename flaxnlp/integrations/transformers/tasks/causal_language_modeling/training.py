from typing import Any, Dict, Tuple, cast

import jax
import optax
from flax.training.common_utils import onehot

from flaxnlp.training.trainer import Trainer, TrainingModule, TrainState

Array = Any
Model = Any


@TrainingModule.register("transformers::causallm")
class TrainingModuleForTransformersCausalLM(TrainingModule):
    @staticmethod
    def loss_fn(logits: Array, labels: Array) -> Array:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(
            shift_logits,
            onehot(shift_labels, shift_logits.shape[-1]),  # type: ignore[no-untyped-call]
        )
        return loss.mean()

    def create_state(
        self,
        rngs: Any,
        trainer: Trainer,
        model: Model,
        inputs: Dict[str, Any],
    ) -> TrainState:
        rngs, dropout_rng = jax.random.split(rngs)
        state = TrainState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.__call__,
            params=model.params,
            tx=trainer.optimizer,
            rngs={"dropout": dropout_rng},
        )
        return cast(TrainState, state)

    def train_step(self, state: TrainState, batch: Dict[str, Any]) -> Tuple[TrainState, Dict[str, Any]]:
        dropout_rng, new_dropout_rng = jax.random.split(state.rngs["dropout"])

        def compute_loss(params: Any) -> Any:
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = self.loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")  # type: ignore[no-untyped-call]

        new_state = state.apply_gradients(grads=grad, rngs={"dropout": new_dropout_rng})  # type: ignore[no-untyped-call]

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")  # type: ignore[no-untyped-call]

        return new_state, metrics

    def eval_step(self, state: TrainState, batch: Dict[str, Any]) -> Dict[str, Any]:
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        loss = self.loss_fn(logits, labels)

        metrics = {"loss": loss, "perplexity": jax.numpy.exp(loss)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")  # type: ignore[no-untyped-call]

        return metrics
