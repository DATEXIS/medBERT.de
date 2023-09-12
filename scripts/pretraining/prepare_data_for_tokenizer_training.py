import ctypes
import os

from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm
from datasets import load_dataset
import re
from multiprocessing import Pool
from char_counter import char_counter
import multiprocessing

contains_japanese = re.compile("[\uac00-\ud7a3]")
contains_chinese = re.compile("[\u3040-\u30ff]")
contains_other = re.compile("[\u4e00-\u9FFF]")


def characters_in_string(characters, string):
    for c in characters:
        if c in string:
            return True
    if contains_japanese.search(string):
        return True
        # japanese
    if contains_chinese.search(string):
        return True
        # chinese
    if contains_other.search(string):
        return True

    return False


def f(x):
    return dataset_as_str.count(x)


HOME = "/home/bressekk/Documents/medbert/"
NUM_PROC = multiprocessing.cpu_count()
if __name__ == "__main__":
    if os.environ["DEBUG"] is not None:
        datasets = load_dataset("GerMedBERT/RefMedIs", use_auth_token=os.getenv("AUTHTOKEN", False))
    else:
        datasets = load_dataset(
            HOME + "datasets/",
            data_files={"train": "mlm_pretraining_data_no_duplicates.csv"},
            use_auth_token=os.getenv("AUTHTOKEN", False)
        )

    datasets["train"] = datasets["train"].filter(
        lambda x: x["text"] is not None and len(x["text"]) > 0 and not x["text"].isspace(),
        num_proc=NUM_PROC
    )

    dataset_as_str = " ".join(datasets["train"]["text"])

    tokenizer = BertTokenizer.from_pretrained("GerMedBERT/medbert-512", use_auth_token=os.getenv("AUTHTOKEN", False))
    vocab = tokenizer.vocab
    vocab = [x.strip() for x in vocab if len(x.strip()) == 1]
    character_appearances = {c: 0 for c in vocab}
    print("Start counting")
    counts = char_counter.count_multi_chars(vocab, dataset_as_str)
    print("Char counting complete")
    character_appearances = {char: count for char, count in zip(vocab, counts)}

    low_frequent_characters = {k: v for k, v in character_appearances.items() if v < 5}
    low_frequent_characters = set(low_frequent_characters.keys())

    datasets["train"] = datasets["train"].filter(
        lambda x: not characters_in_string(low_frequent_characters, x["text"]), num_proc=NUM_PROC
    )
    print("Filtered out low frequent chars")

    datasets.save_to_disk(HOME + "datasets/tokenizer_ready_dataset")
    print("Saved prepared dataset to disk")