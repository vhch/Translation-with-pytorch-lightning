import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
    WMT16TranslationDataModule,
)
from pytorch_lightning.loggers import WandbLogger
import argparse
from datetime import datetime
import torch
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler
import os

CUDA_LAUNCH_BLOCKING=1

torch.set_num_threads(16)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
now = datetime.now()
wandb_logger = WandbLogger(name=f'no byte level', project='translation')


if __name__ == "__main__":
    pl.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=32, type=int,
                        help='number of each process batch number')
    parser.add_argument('-n', '--mname', default="facebook/mbart-large-cc25", type=str,
                        help='model name in huggingface')
    parser.add_argument('-d', '--dataset', default="facebook/mbart-large-cc25", type=str,
                        help='model name in huggingface')
    args = parser.parse_args()

    mname = args.mname

    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="de_DE")
    model = TranslationTransformer(
        pretrained_model_name_or_path=mname,
        val_target_max_length=128,
        num_beams=5,
        compute_generate_metrics=True,
        # load_weights=False,
        lr=5e-4,
        warmup_steps=0.04,
        batch_size=args.batch
    )
    dm = WMT16TranslationDataModule(
        # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        dataset_name='bbaaaa/iwslt14-de-en-preprocess',
        dataset_config_name="de-en",
        source_language="de",
        target_language="en",
        max_source_length=128,
        max_target_length=128,
        padding="max_length",
        tokenizer=tokenizer,
        batch_size=args.batch,
        num_workers=16,
        model=model.model
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        monitor='val_bleu_score',
        mode='max',
        save_last=True,
        # every_n_train_steps=2000,
    )

    trainer = pl.Trainer(
        # fast_dev_run=True,
        logger=wandb_logger,
        accelerator="auto",
        # accelerator="cpu",
        devices=[0, 1, 2, 3],
        max_epochs=100,
        strategy='ddp',
        precision=16,
        limit_val_batches=0.05,
        # callbacks=[checkpoint_callback, early_stop_callback],
        callbacks=[checkpoint_callback],
        # accumulate_grad_batches=8,
    )

    wandb_logger.watch(model, log="all")

    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path='best')
