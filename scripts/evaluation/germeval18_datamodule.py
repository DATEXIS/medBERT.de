import os

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from radiology.chest_ct_datamodule import ClassificationDataset
from sklearn import preprocessing
from transformers import AutoTokenizer, DataCollatorWithPadding

from datasets import load_dataset


class Germeval18Datamodule(pl.LightningDataModule):
    # 2 classes
    def __init__(self, batch_size=8, num_workers=0, tokenizer_name: str = "bert-base-german-cased"):
        super().__init__()
        data = load_dataset("philschmid/germeval18")
        train_valid = data["train"].train_test_split(test_size=0.1)
        self.valid_data = train_valid["test"]
        self.train_data = train_valid["train"]
        self.test_data = data["test"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_auth_token=os.getenv("AUTHTOKEN", False), use_fast=True
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        labelencoder = preprocessing.LabelEncoder()
        labelencoder.fit(self.train_data["binary"])

        self.train_texts = self.tokenizer(self.train_data["text"], truncation=True, max_length=512).input_ids
        self.training_labels = labelencoder.transform(self.train_data["binary"])
        self.test_texts = self.tokenizer(self.test_data["text"], truncation=True, max_length=512).input_ids
        self.test_labels = labelencoder.transform(self.test_data["binary"])
        self.valid_texts = self.tokenizer(self.valid_data["text"], truncation=True, max_length=512).input_ids
        self.val_labels = labelencoder.transform(self.valid_data["binary"])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=64, max_length=512)
        training_dataset = ClassificationDataset(
            {"text": self.train_texts, "labels": self.training_labels}, multilabel=False
        )
        return torch.utils.data.DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=64, max_length=512)
        test_dataset = ClassificationDataset({"text": self.test_texts, "labels": self.test_labels}, multilabel=False)
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=64, max_length=512)
        dev_dataset = ClassificationDataset({"text": self.valid_texts, "labels": self.val_labels}, multilabel=False)
        return torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    module = Germeval18Datamodule()
    a = 0
