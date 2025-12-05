import os
import json
import re
from glob import glob
from collections import Counter

# simple tokenizer
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def basic_tokenize(text: str):
    text = text.lower()
    return TOKEN_PATTERN.findall(text)


# def label_iterator(json_dir: str):
#     """
#     Iterate over tokenized labels in all JSON files.
#     Supports:
#       - file contains a single sample dict
#       - file contains a list of sample dicts (batches)
#     """
#     json_paths = sorted(glob(os.path.join(json_dir, "*.json")))
#     if not json_paths:
#         print(f"[label_iterator] WARNING: no *.json files found in {json_dir}")
#
#     for path in json_paths:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         # Case 1: one sample per file
#         if isinstance(data, dict):
#             label = data["label"]
#             yield basic_tokenize(label)
#
#         # Case 2: batch file: list of samples
#         elif isinstance(data, list):
#             for sample in data:
#                 if not isinstance(sample, dict):
#                     continue
#                 if "label" not in sample:
#                     continue
#                 label = sample["label"]
#                 yield basic_tokenize(label)
#
#         else:
#             continue


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