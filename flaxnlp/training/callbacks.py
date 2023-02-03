import typing
from typing import Any, Dict

from flaxnlp.training.exceptions import StopEarly

if typing.TYPE_CHECKING:
    from flaxnlp.training.train_state import TrainState
    from flaxnlp.training.trainer import Trainer


class Callback:
    def on_start(
        self,
        trainer: "Trainer",
        train_state: "TrainState",
    ) -> None:
        pass

    def on_batch(
        self,
        trainer: "Trainer",
        train_state: "TrainState",
        batch_inputs: Dict[str, Any],
        batch_outputs: Dict[str, Any],
        epoch: int,
        is_training: bool,
    ) -> None:
        pass

    def on_epoch(
        self,
        trainer: "Trainer",
        train_state: "TrainState",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_end(
        self,
        trainer: "Trainer",
        train_state: "TrainState",
    ) -> None:
        pass


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int = 3,
        metric: str = "-val_loss",
    ):
        self.patience = patience
        self.metric = metric[1:] if metric.startswith(("-", "+")) else metric
        self.direction = -1 if self.metric.startswith("-") else 1
        self.best_metric = float("inf")
        self.counter = 0

    def on_epoch(
        self,
        trainer: "Trainer",
        train_state: "TrainState",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        if self.direction * metrics[self.metric] > self.direction * self.best_metric:
            self.best_metric = metrics[self.metric]
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            raise StopEarly
