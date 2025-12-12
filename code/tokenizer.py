import re
from collections import Counter

"""
We use the simplest counter-based tokenizer to avoid extra dependencies
and possibly more efficient inference, time-wise.
"""
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def basic_tokenize(text: str):
    text = text.lower()
    return TOKEN_PATTERN.findall(text)

def build_vocab_from_sentences(sentences, min_freq = 1) :
    counter = Counter()
    for sent in sentences:
        tokens = basic_tokenize(sent)
        counter.update(tokens)

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab