from typing import Any, List, Optional

import torch

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.seq2seq.utils import _pad_tensors_to_max_len


class Seq2SeqTransformer(TaskTransformer):
    def __init__(
        self,
        *args,
        val_target_max_length: Optional[int] = 128,
        num_beams: Optional[int] = 1,
        compute_generate_metrics: bool = True,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.val_target_max_length = val_target_max_length
        self.num_beams = num_beams
        self.should_compute_generate_metrics = compute_generate_metrics

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        labels = batch['labels']
        outputs = self.model(**batch)
        logits = outputs[1]
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = criterion(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        self.log("train_loss", loss)
        # outputs = self.model(**batch)
        # loss = outputs[0]
        # self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        if self.should_compute_generate_metrics:
            self.compute_generate_metrics(batch, prefix)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("test", batch)

    def compute_generate_metrics(self, batch, prefix):
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        max_length = self.val_target_max_length if self.val_target_max_length else self.model.config.max_length
        num_beams = self.num_beams if self.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams,
                # length_penalty=0.6
                length_penalty=1.0
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = _pad_tensors_to_max_len(
                model_cfg=self.model.config, tensor=generated_tokens, max_length=max_length
            )
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str

    def tokenize_labels(self, labels: torch.Tensor) -> List[str]:
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return [str.strip(s) for s in label_str]
