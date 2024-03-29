from datasets import load_dataset

dataset = load_dataset("bbaaaa/iwslt14-de-en-preprocess", 'de-en', split='train')

def convert_to_features(
    examples
):
    examples['tokenizers'] = [ex['de'] + ' ' + ex['en'] for ex in examples["translation"]]
    return examples

dataset = dataset.map(convert_to_features, batched=True)

from tokenizers import Tokenizer
# tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer = Tokenizer.from_pretrained("flaubert/flaubert_base_cased")

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=8000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])  # Adding [BOS] and [EOS] here

tokenizer.train_from_iterator(dataset['tokenizers'], trainer)

tokenizer.save("tokenizer.json")
tokenizer.model.save(".")
