import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, DataCollatorWithPadding


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, multilabel: bool = True):
        self.data = data
        self.multilabel = multilabel

    def __len__(self):
        return len(self.data["text"])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data["text"][idx])
        attention_mask = torch.ones(len(input_ids), dtype=torch.bool)
        labels = torch.tensor(self.data["labels"][idx])
        if self.multilabel:
            labels = labels.float()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_filename: str = "/Users/pgrundmann/Downloads/radiologie-benchmarks/chest-ct/ct_reports.csv",
        test_filename: str = "/Users/pgrundmann/Downloads/radiologie-benchmarks/chest-ct/ct_reports_test.csv",
        val_filename: str = "/pvc/radiologie-benchmarks/chest-ct/val.csv",
        batch_size=8,
        num_workers=0,
        tokenizer_name: str = "bert-base-german-cased",
    ):
        super().__init__()
        #df = pd.read_csv(train_filename)
        #df_tmp = pd.read_csv(test_filename)
        #df = pd.concat([df, df_tmp], ignore_index=True)

        #index_split = self.read_pickle()
        #self.training_data = df[df.index.isin(index_split["train_idx"])]
        #elf.dev_data = df[df.index.isin(index_split["val_idx"])]
        #self.test_data = df[df.index.isin(index_split["test_idx"])]

        self.training_data = pd.read_csv(train_filename)
        self.test_data = pd.read_csv(test_filename)
        self.dev_data = pd.read_csv(val_filename)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_auth_token=os.getenv("AUTHTOKEN", False), use_fast=True
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def save_pickle(self, df):
        import pickle

        df_train, df_val, df_test = np.split(
            df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
        )
        train_idx = list(df_train.index)
        val_idx = list(df_val.index)
        test_idx = list(df_test.index)
        index_split = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        with open("evaluation/radiology/chest_ct_index_split.pickle", "wb") as handle:
            pickle.dump(index_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_pickle(self):
        import pickle

        with open("evaluation/radiology/chest_ct_index_split.pickle", "rb") as handle:
            index_split = pickle.load(handle)
        return index_split

    def prepare_data(self) -> None:
        self.training_data = self.training_data[self.training_data["labels"].notna()]
        self.test_data = self.test_data[self.test_data["labels"].notna()]
        self.dev_data = self.dev_data[self.dev_data["labels"].notna()]

        self.training_data["labels"] = self.training_data["labels"].map(lambda x: x.split(","))
        self.dev_data["labels"] = self.dev_data["labels"].map(lambda x: x.split(","))
        self.test_data["labels"] = self.test_data["labels"].map(lambda x: x.split(","))
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(
            pd.concat([self.training_data["labels"], self.dev_data["labels"], self.test_data["labels"]]).tolist()
        )
        self.training_data["labels"] = mlb.transform(self.training_data["labels"]).tolist()
        self.dev_data["labels"] = mlb.transform(self.dev_data["labels"]).tolist()
        self.test_data["labels"] = mlb.transform(self.test_data["labels"]).tolist()

        self.training_data["text"] = self.tokenizer(
            self.training_data["text"].to_list(), truncation=True, max_length=512
        ).input_ids

        self.dev_data["text"] = self.tokenizer(
            self.dev_data["text"].to_list(), truncation=True, max_length=512
        ).input_ids

        self.test_data["text"] = self.tokenizer(
            self.test_data["text"].to_list(), truncation=True, max_length=512
        ).input_ids

    def prepare_data_old(self) -> None:
        self.training_data = self.training_data[self.training_data["labels"].notna()]
        self.test_data = self.test_data[self.test_data["labels"].notna()]
        self.training_data["labels"] = self.training_data["labels"].map(lambda x: x.split(","))
        self.test_data["labels"] = self.test_data["labels"].map(lambda x: x.split(","))
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(pd.concat([self.training_data["labels"], self.test_data["labels"]]).tolist())
        self.training_data["labels"] = mlb.transform(self.training_data["labels"]).tolist()
        self.test_data["labels"] = mlb.transform(self.test_data["labels"]).tolist()

        self.training_data["text"] = self.tokenizer(
            self.training_data["text"].to_list(), truncation=True, max_length=512
        ).input_ids

        self.dev_data = self.training_data.sample(frac=0.3, random_state=42)
        self.training_data = self.training_data.drop(self.dev_data.index)

        self.test_data["text"] = self.tokenizer(
            self.test_data["text"].to_list(), truncation=True, max_length=512
        ).input_ids

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=64, max_length=512)
        training_dataset = ClassificationDataset(
            {"text": self.training_data["text"].tolist(), "labels": self.training_data["labels"].tolist()}
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
        test_dataset = ClassificationDataset(
            {"text": self.test_data["text"].tolist(), "labels": self.test_data["labels"].tolist()}
        )
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
        dev_dataset = ClassificationDataset(
            {"text": self.dev_data["text"].tolist(), "labels": self.dev_data["labels"].tolist()}
        )
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
    datamodule = CTDataModule(
                            train_filename="/pvc/radiologie-benchmarks/chest-ct/train.csv",
                            test_filename="/pvc/radiologie-benchmarks/chest-ct/test.csv",
                            val_filename = "/pvc/radiologie-benchmarks/chest-ct/val.csv",
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