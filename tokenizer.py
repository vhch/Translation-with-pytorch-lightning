from datasets import load_dataset

dataset = load_dataset("wmt14", 'de-en', split='train')


def convert_to_features(
    examples
):
    examples['tokenizers'] = [ex['de'] + ex['en'] for ex in examples["translation"]]
    return examples

dataset = dataset.map(convert_to_features, batched=True)

from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=52000, special_tokens=["[BOS]", "[PAD]", "[EOS]", "[UNK]", "[MASK]"])  # Adding [BOS] and [EOS] here

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
# tokenizer.pre_tokenizer = ByteLevel()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), ByteLevel()])

from tokenizers import decoders
tokenizer.decoder = decoders.ByteLevel()

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[("[BOS]", 0), ("[EOS]", 2)],
)

tokenizer.train_from_iterator(dataset['tokenizers'], trainer)

tokenizer.save("tokenizer.json")
tokenizer.model.save(".")
