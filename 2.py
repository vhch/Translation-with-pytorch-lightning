from transformers import AutoTokenizer
from lightning_transformers.task.nlp.translation import TranslationTransformer
import torch

model = TranslationTransformer(
    pretrained_model_name_or_path="google/mt5-base",
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="google/mt5-base"),
    load_weights=False,
)


print(model.hf_predict("¡Hola Sean!"))

model.load_from_checkpoint("/sj/test/lightning_logs/version_4/checkpoints/epoch=0-step=9537.ckpt")

print("load")

print(model.hf_predict("¡Hola Sean!"))
