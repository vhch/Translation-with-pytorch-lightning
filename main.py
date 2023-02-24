import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
    WMT16TranslationDataModule,
)
from pytorch_lightning.loggers import WandbLogger
import argparse
from datetime import datetime

now = datetime.now()
# wandb_logger = WandbLogger(name=f'{now.date()}-transformer-base', project='translation-wmt14')
wandb_logger = WandbLogger(name=f'bart-base768-batch512-epoch100', project='translation-wmt14')


if __name__ == "__main__":
    pl.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='number of each process batch number')
    args = parser.parse_args()

    mname = "bbaaaa/myfork"
    # mname = "bbaaaa/myfork2"

    # tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = TranslationTransformer(
        pretrained_model_name_or_path=mname,
        val_target_max_length=128,
        num_beams=5,
        compute_generate_metrics=True,
        load_weights=False,
        lr=5e-4,
        warmup_steps=0.01,
        batch_size=args.batch
    )
    dm = WMT16TranslationDataModule(
        # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        dataset_name='wmt14',
        dataset_config_name="de-en",
        source_language="en",
        target_language="de",
        max_source_length=128,
        max_target_length=128,
        padding="max_length",
        tokenizer=tokenizer,
        batch_size=args.batch,
        # num_workers=12,
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
        # devices=[4, 5, 6, 7],
        # max_epochs=15,
        max_epochs=100,
        strategy='ddp',
        # strategy='deepspeed_stage_2',
        precision=16,
        # limit_train_batches=0.05,
        # callbacks=[checkpoint_callback, early_stop_callback],
        callbacks=[checkpoint_callback],
        # accumulate_grad_batches=8,
    )

    wandb_logger.watch(model, log="all")

    trainer.fit(model, dm)
    trainer.test(model, dm)
    # trainer.test(model, dm, ckpt_path='/sj/test/translation-wmt14/3nfrb4fc/checkpoints/epoch=38-step=42900.ckpt')

    # trainer = pl.Trainer(accelerator='gpu', devices=[0])
    # trainer.validate(model, dm)
