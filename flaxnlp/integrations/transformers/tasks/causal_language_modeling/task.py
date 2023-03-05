from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import colt
import jax

from flaxnlp.models import Model
from flaxnlp.tasks import Task
from flaxnlp.training import Trainer

from .datamodule import DataModuleForTransformersCausalLM


@Task.register("transformers::train_causallm")
class TaskForTransformersCausalLM(Task):
    def __init__(
        self,
        model: colt.Lazy[Model],
        trainer: Trainer,
        datamodule: DataModuleForTransformersCausalLM,
        train_dataset: Union[Sequence[Mapping[str, Any]], Sequence[str]],
        val_dataset: Optional[Union[Sequence[Mapping[str, Any]], Sequence[str]]] = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.trainer = trainer
        self.datamodule = datamodule
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.random_seed = random_seed

    def run(self, work_dir: Union[str, PathLike]) -> None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        model = self.model.construct(**self.datamodule.get_model_kwargs())
        train_dataset = self.datamodule.read_dataset(self.train_dataset)
        val_dataset = self.datamodule.read_dataset(self.val_dataset) if self.val_dataset else None
        state = self.trainer.train(
            rngs=jax.random.PRNGKey(self.random_seed),
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        model.save_pretrained(work_dir / "model", params=state.params)
