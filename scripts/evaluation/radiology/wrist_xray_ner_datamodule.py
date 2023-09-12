import json
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import AutoTokenizer, BertTokenizerFast


def load_json_pandas(file_path):
    data = list()
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame.from_records(data)


def get_all_entities(df):
    entity_ranges = df.label.tolist()
    flat_list = []

    for ents in entity_ranges:
        for ent in ents:
            flat_list.append(ent)

    ents_set = list(set([ent[-1] for ent in flat_list]))
    ents_set.append("O")
    return ents_set


def entity_range_builder(doc_entity_ranges, text_tokenized):
    offset_mapping = text_tokenized["offset_mapping"]
    label_dict = dict()

    for index, doc in enumerate(doc_entity_ranges):

        if index not in label_dict:
            label_dict[index] = dict()

        for i, off in enumerate(offset_mapping[index]):
            for entity_range in doc:

                start = entity_range[0]
                end = entity_range[1]
                entity = entity_range[2]

                if off[0] >= start and off[1] <= end:
                    label_dict[index][i] = entity

                elif off[0] == 0 and off[1] == 0:
                    label_dict[index][i] = None

            if i not in label_dict[index]:
                label_dict[index][i] = "O"

        assert len(label_dict[index]) == 512
    return label_dict


def create_label_mappings(ents_set):

    ents_set = sorted(ents_set)
    ent_to_index = dict([(ent, i) for i, ent in enumerate(ents_set)])
    index_to_ent = dict([(i, ent) for i, ent in enumerate(ents_set)])

    return ent_to_index, index_to_ent


def tokenize_text(df, model):

    all_texts = df.text.tolist()
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, use_auth_token=os.getenv("AUTHTOKEN", False))

    text_tokenized = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=all_texts,  # Preprocess sentence
        add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
        max_length=512,  # Max length to truncate/pad
        pad_to_max_length=True,  # Pad sentence to max length
        return_attention_mask=True,  # Return attention mask
        return_offsets_mapping=True,
        truncation=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )
    return text_tokenized


def prepare_NER_dataset(label_dict, ent_to_index, text_tokenized):
    docs_labels = list()
    word_masks = list()

    for i, doc in enumerate(label_dict):
        docs_labels.append([])
        word_masks.append([])
        for tok in label_dict[doc]:
            label_encoding = ent_to_index.get(label_dict[doc][tok], -100)
            docs_labels[i].append(label_encoding)
        previous_word_idx = None
        for word_idx in text_tokenized.word_ids(i):
            if word_idx != None and previous_word_idx != word_idx:
                word_masks[i].append(word_idx)
            else:
                word_masks[i].append(-100)
            previous_word_idx = word_idx
    text_tokenized["labels"] = docs_labels
    return text_tokenized, docs_labels, word_masks


class NERClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df, which_tokenizer, all_label_entities):

        self.text_tokenized = tokenize_text(df, which_tokenizer)
        entity_ranges = df.label.tolist()
        label_dict = entity_range_builder(doc_entity_ranges=entity_ranges, text_tokenized=self.text_tokenized)

        ent_to_index, _ = create_label_mappings(all_label_entities)
        _, labels, word_masks = prepare_NER_dataset(label_dict, ent_to_index, self.text_tokenized)

        self.labels = labels
        self.word_masks = word_masks

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        item = {key: val[idx] for key, val in self.text_tokenized.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["word_mask"] = torch.tensor(self.word_masks[idx])
        return item


class NERDataModule(pl.LightningDataModule):
    # 9 classes
    def __init__(
        self,
        filename: str = "/home/neuron/PycharmProjects/data/radiologie-benchmarks/wrist-xray-and-ct/scaphiod-annotiert-1.jsonl",
        train_filename = "", 
        val_filename = "", 
        test_filename = "",
        batch_size=8,
        num_workers=0,
        tokenizer_name: str = "bert-base-german-cased",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_auth_token=os.getenv("AUTHTOKEN", False), use_fast=True
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        #df = load_json_pandas(filename)

        #index_split = self.read_pickle()
        #df_train = df[df.index.isin(index_split["train_idx"])]
        #df_val = df[df.index.isin(index_split["val_idx"])]
        #df_test = df[df.index.isin(index_split["test_idx"])]
        df_train = pd.read_csv(train_filename)
        df_test = pd.read_csv(test_filename)
        df_val = pd.read_csv(val_filename)

        df_train['label'] = df_train['label'].apply(eval)
        df_val['label'] = df_val['label'].apply(eval)
        df_test['label'] = df_test['label'].apply(eval)

        df = pd.concat([df_train, df_test, df_val]).reset_index(drop=True)
        ents_set = get_all_entities(df)
        self.train_dataset = NERClassificationDataset(
            df=df_train, which_tokenizer=tokenizer_name, all_label_entities=ents_set
        )
        self.val_dataset = NERClassificationDataset(
            df=df_val, which_tokenizer=tokenizer_name, all_label_entities=ents_set
        )
        self.test_dataset = NERClassificationDataset(
            df=df_test, which_tokenizer=tokenizer_name, all_label_entities=ents_set
        )

    def save_pickle(self, df):
        import pickle

        df_train, df_val, df_test = np.split(
            df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
        )
        train_idx = list(df_train.index)
        val_idx = list(df_val.index)
        test_idx = list(df_test.index)
        index_split = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        with open("evaluation/radiology/wrist_xray_ner_index_split.pickle", "wb") as handle:
            pickle.dump(index_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_pickle(self):
        import pickle

        with open("evaluation/radiology/wrist_xray_ner_index_split.pickle", "rb") as handle:
            index_split = pickle.load(handle)
        return index_split

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":

    base_path = "/home/neuron/PycharmProjects/data/radiologie-benchmarks/wrist-xray-and-ct/"
    file_path = f"{base_path}/scaphiod-annotiert-1.jsonl"
    model_name = "deepset/gbert-base"

    datamodule = NERDataModule(train_filename="/pvc/radiologie-benchmarks/wrist-xray-and-ct/train.csv",
                                val_filename="/pvc/radiologie-benchmarks/wrist-xray-and-ct/val.csv", 
                                test_filename="/pvc/radiologie-benchmarks/wrist-xray-and-ct/test.csv",
                                batch_size=8, 
                                num_workers=4, 
                                tokenizer_name=model_name)


