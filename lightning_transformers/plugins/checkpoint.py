import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pytorch_lightning.plugins import TorchCheckpointIO

from lightning_transformers.core import TaskTransformer


class HFSaveCheckpoint(TorchCheckpointIO):
    """Allows you to save an additional HuggingFace Hub compatible checkpoint."""

    def __init__(self, model: TaskTransformer, suffix: str = "_huggingface"):
        self._model = model
        self._suffix = suffix

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        super().save_checkpoint(checkpoint, path, storage_options)
        base_path = os.path.splitext(path)[0] + self._suffix
        self._model.save_hf_checkpoint(base_path)
