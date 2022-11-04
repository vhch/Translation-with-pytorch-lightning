import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
    WMT16TranslationDataModule,
)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/mt5-base"
    )
    model = TranslationTransformer(
        pretrained_model_name_or_path="google/mt5-base",
        n_gram=4,
        smooth=False,
        val_target_max_length=142,
        num_beams=None,
        compute_generate_metrics=True,
    )
    dm = WMT16TranslationDataModule(
        # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        dataset_config_name="ro-en",
        source_language="en",
        target_language="ro",
        max_source_length=64,
        max_target_length=64,
        padding="max_length",
        tokenizer=tokenizer,
    )

    trainer = pl.Trainer(accelerator="auto", devices=[0, 1, 2, 3], max_epochs=1)

    trainer.fit(model, dm)
