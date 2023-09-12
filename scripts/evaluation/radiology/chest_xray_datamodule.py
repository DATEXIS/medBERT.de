import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from radiology.chest_ct_datamodule import ClassificationDataset
from transformers import AutoTokenizer, DataCollatorWithPadding


class XRayDataModule(pl.LightningDataModule):
    # 9 classes
    def __init__(
        self,
        train_filename: str = "/Users/pgrundmann/Downloads/radiologie-benchmarks/chest-xray/train.csv",
        test_filename: str = "/Users/pgrundmann/Downloads/radiologie-benchmarks/chest-xray/test.csv",
        val_filename: str = "/Users/pgrundmann/Downloads/radiologie-benchmarks/chest-xray/valid.csv",
        batch_size=8,
        num_workers=0,
        tokenizer_name: str = "bert-base-german-cased",
    ):
        super().__init__()
        self.training_data = pd.read_csv(train_filename)
        self.test_data = pd.read_csv(test_filename)
        self.val_data = pd.read_csv(val_filename)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_auth_token=os.getenv("AUTHTOKEN", False), use_fast=True
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.training_labels = self.training_data.loc[:, self.training_data.columns != "text"].to_numpy()
        self.training_texts = self.tokenizer(
            self.training_data["text"].tolist(), truncation=True, max_length=512
        ).input_ids

        self.test_labels = self.test_data.loc[:, self.test_data.columns != "text"].to_numpy()
        self.test_texts = self.tokenizer(self.test_data["text"].tolist(), truncation=True, max_length=512).input_ids

        self.val_labels = self.val_data.loc[:, self.val_data.columns != "text"].to_numpy()
        self.val_texts = self.tokenizer(self.val_data["text"].tolist(), truncation=True, max_length=512).input_ids

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=64, max_length=512)
        training_dataset = ClassificationDataset({"text": self.training_texts, "labels": self.training_labels})
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
        test_dataset = ClassificationDataset({"text": self.test_texts, "labels": self.test_labels})
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
        dev_dataset = ClassificationDataset({"text": self.val_texts, "labels": self.val_labels})
        return torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )

if __name__ == '__main__':
    datamodule = XRayDataModule(
                            train_filename="/pvc/radiologie-benchmarks/chest-xray/train.csv",
                            test_filename="/pvc/radiologie-benchmarks/chest-xray/test.csv",
                            val_filename = "/pvc/radiologie-benchmarks/chest-xray/val.csv",
                            batch_size=8,
                            num_workers=0,
                            tokenizer_name= "bert-base-german-cased",
                        )
    from pytorch_lightning import Trainer
    trainer = Trainer(
        max_epochs=100,
        enable_checkpointing=False,
        accumulate_grad_batches=1,
    )

    from classification_model import ClassificationModel
    model = ClassificationModel(
        modelname="bert-base-german-cased", lr=0.1, warmup_steps=1, num_classes=24
    )
    trainer.fit(model, datamodule)