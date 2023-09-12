import os

import pytorch_lightning as pl
import torch

from transformers import BertTokenizerFast, AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset, Sequence, ClassLabel

GGPONC_LABELS = [
    "Other_Finding",
    "Diagnosis_or_Pathology",
    "Therapeutic",
    "Diagnostic",
    "Nutrient_or_Body_Substance",
    "External_Substance",
    "Clinical_Drug",
]

MAX_LENGTH = 512


def tokenize_and_align_labels(tokenizer, examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    labels = []
    word_masks = []
    for i, label in enumerate(examples["_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        word_mask = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            word_mask_idx = -100
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                word_mask_idx = word_idx
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            word_mask.append(word_mask_idx)
            previous_word_idx = word_idx

        labels.append(label_ids)
        word_masks.append(word_mask)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_mask"] = word_masks

    return tokenized_inputs


class GGPONCNERDataModule(pl.LightningDataModule):

    KEYS = ["input_ids", "attention_mask", "offset_mapping", "labels", "word_mask"]

    def __init__(
        self, folder_name: str, batch_size=8, num_workers=0, tokenizer_name: str = "bert-base-german-cased",
    ):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_auth_token=os.getenv("AUTHTOKEN", False), use_fast=True, add_prefix_space=True
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.train_dataset = self.load_dataset(os.path.join(folder_name, "train_fine_long.json"))
        self.val_dataset = self.load_dataset(os.path.join(folder_name, "dev_fine_long.json"))
        self.test_dataset = self.load_dataset(os.path.join(folder_name, "test_fine_long.json"))

    def load_dataset(self, ggponc_split, label_all_tokens=True):
        dataset = load_dataset("json", data_files=ggponc_split)["train"]
        features = dataset.features

        tag2idx = {}
        labels = ["O"] + GGPONC_LABELS
        for i, tag in enumerate(labels):
            if tag == "O":
                tag2idx[tag] = i
            else:
                tag2idx["B-" + tag] = i
                tag2idx["I-" + tag] = i
        dataset = dataset.map(lambda e: {"_tags": [tag2idx[t] for t in e["tags"]]})
        features["_tags"] = Sequence(ClassLabel(num_classes=len(labels), names=(labels)))

        dataset = dataset.cast(features)
        dataset = dataset.map(lambda e: tokenize_and_align_labels(self.tokenizer, e, label_all_tokens), batched=True, load_from_cache_file=False)
        return dataset.remove_columns([k for k in dataset.column_names if k not in self.KEYS])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )
