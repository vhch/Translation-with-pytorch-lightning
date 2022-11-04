from transformers import AutoTokenizer
from lightning_transformers.task.nlp.translation import TranslationTransformer

model = TranslationTransformer(
    pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random",
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random"),
)
model.hf_predict("¡Hola Sean!")
