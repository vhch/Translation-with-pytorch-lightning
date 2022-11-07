import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
    WMT16TranslationDataModule,
)
from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(name='test-11-07', project='translation')
wandb_logger = WandbLogger(project='translation')


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path="google/mt5-base"
    # )
    tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
    model = TranslationTransformer(
        # pretrained_model_name_or_path="google/mt5-base",
        pretrained_model_name_or_path="google/bert2bert_L-24_wmt_en_de",
        n_gram=1,
        smooth=False,
        val_target_max_length=128,
        num_beams=4,
        compute_generate_metrics=True,
        load_weights=False,
    )
    dm = WMT16TranslationDataModule(
        # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        dataset_config_name="ro-en",
        source_language="en",
        target_language="ro",
        max_source_length=128,
        max_target_length=128,
        padding="max_length",
        tokenizer=tokenizer,
        batch_size=24,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5, monitor='val_loss', mode='min')

    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        accelerator="auto",
        devices=[0, 1, 2, 3],
        max_epochs=10,
        strategy='ddp',
        precision=16,
        # limit_train_batches=0.05,
        callbacks=early_stop_callback,
    )
    trainer.fit(model, dm)

    # trainer = pl.Trainer(accelerator='gpu', devices=[0])
    # trainer.validate(model, dm)
