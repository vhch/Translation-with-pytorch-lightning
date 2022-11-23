# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Type

import torch
import transformers
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.core import TaskTransformer


class LanguageModelingTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Language Modeling Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForCausalLM``)
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self, *args, downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForCausalLM, **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"

    def generate(self, text: str, device: torch.device = torch.device("cpu"), **kwargs) -> Any:
        if self.tokenizer is None:
            raise MisconfigurationException(
                "A tokenizer is required to use the `generate` function. "
                "Please pass a tokenizer `LanguageModelingTransformer(tokenizer=...)`."
            )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        return self.model.generate(inputs["input_ids"], **kwargs)