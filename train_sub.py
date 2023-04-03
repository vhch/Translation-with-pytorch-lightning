from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, get_cosine_schedule_with_warmup, get_scheduler, get_polynomial_decay_schedule_with_warmup
import evaluate
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

dataset = load_dataset('bbaaaa/iwslt14-de-en-preprocess')
d = dataset['train']

def convert_to_features(
    examples
):
    examples['tokenizers'] = [ex['de'] + ' ' + ex['en'] for ex in examples["translation"]]
    return examples

d = d.map(convert_to_features, batched=True)

model_checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenizer = tokenizer.train_new_from_iterator(d['tokenizers'], 16000)

max_input_length = 128
max_target_length = 128
src_text_column_name = 'de'
tgt_text_column_name = 'en'
padding = 'longest'

def preprocess_function(
    examples: Any,
    tokenizer: PreTrainedTokenizerBase,
    padding: str,
    max_source_length: int,
    max_target_length: int,
    src_text_column_name: str,
    tgt_text_column_name: str,
):
    inputs = [ex[src_text_column_name] for ex in examples["translation"]]
    targets = [ex[tgt_text_column_name] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


preprocess_function = partial(
    preprocess_function,
    tokenizer=tokenizer,
    padding=padding,
    max_source_length=max_input_length,
    max_target_length=max_target_length,
    src_text_column_name=src_text_column_name,
    tgt_text_column_name=tgt_text_column_name,
)

dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

model_checkpoint = "bbaaaa/fsmt"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_config(config)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = evaluate.load("sacrebleu")

train_dataloader = DataLoader(
    dataset["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=128,
)
eval_dataloader = DataLoader(
    dataset["validation"],
    collate_fn=data_collator,
    batch_size=128
)
test_dataloader = DataLoader(
    dataset["test"],
    collate_fn=data_collator,
    batch_size=128
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4)
accelerator = Accelerator(mixed_precision='fp16')
model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader
)

num_train_epochs = 100
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
num_warmup_steps = num_training_steps * 0.04
max_grad_norm = 0.0

lr_scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    power=2.0,
)
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_training_steps,
# )


def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # 학습
    model.train()
    for batch in tqdm(train_dataloader):
        outputs = model(**batch)
        # loss = outputs.loss
        logits = outputs[1]
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = criterion(logits.view(-1, model.config.vocab_size), batch['labels'].view(-1))
        accelerator.backward(loss)

        # accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # 평가
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=5,
                length_penalty=1.0,
            )
        labels = batch["labels"]

        # 예측과 레이블을 모으기 전에 함께 패딩 수행
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute(tokenize='none')
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")


# 평가
model.eval()
for batch in tqdm(test_dataloader):
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=5,
            length_penalty=1.0,
        )
    labels = batch["labels"]

    # 예측과 레이블을 모으기 전에 함께 패딩 수행
    generated_tokens = accelerator.pad_across_processes(
        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

    predictions_gathered = accelerator.gather(generated_tokens)
    labels_gathered = accelerator.gather(labels)

    decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
    metric.add_batch(predictions=decoded_preds, references=decoded_labels)

results = metric.compute(tokenize='none')
print(f"Test BLEU score: {results['score']:.2f}")
