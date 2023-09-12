from itertools import chain

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import BertTokenizerFast
import torch.utils.data
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from datasets import load_dataset, load_from_disk


class SimpleMLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["attention_mask"] = torch.tensor(sample["attention_mask"])
        sample["attention_mask"][sample["attention_mask"] > 0] = 1
        sample["attention_mask"] = sample["attention_mask"].tolist()
        return sample


class MLMDataModule(pl.LightningDataModule):
    def __init__(self,
                 datafiles,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 tokenizer_name: str = "/home/bressekk/Documents/medbert/bert-base-medical-german",
                 cache_path="../../cache",
                 max_seq_len: int = 128,
                 num_proc: int = 32, 
                 dont_overwrite_existing_dataset: bool = True):
        super().__init__()
        self.datafiles = datafiles
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.cache_path = os.path.abspath(cache_path)
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, pad_to_multiple_of=64, mlm_probability=0.15)
        self.num_proc = num_proc
        self.dont_overwrite_existing_dataset = dont_overwrite_existing_dataset

    def setup(self, stage, **kwargs):
        self.train_dataset = SimpleMLMDataset(load_from_disk(self.cache_path + "/train"))
        self.val_dataset = SimpleMLMDataset(load_from_disk(self.cache_path + "/test"))

    def prepare_data(self, **kwargs):

        if self.dont_overwrite_existing_dataset:  
            return 
        
        shutil.rmtree(self.cache_path)
        
        # 1. load all data
        datasets = load_dataset("csv", data_files = self.datafiles)

        # 2. Define tokenizer and grouping function
        def tokenize_function(examples):

            examples['text'] = [
                line for line in examples['text'] if line is not None and len(line) > 0 and not line.isspace()
            ]
            return self.tokenizer(
                examples["text"],
                padding=False,
                truncation=False,
                add_special_tokens=False
                )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, 
            # you can customize this part to your needs.
            if total_length >= self.max_seq_len:
                total_length = (total_length // self.max_seq_len) * self.max_seq_len
            # Split by chunks of max_len.
            result = {
                k: [
                    (
                        [self.tokenizer.cls_token_id] 
                        + t[i: i + self.max_seq_len - 2] 
                        + [self.tokenizer.sep_token_id]
                        ) for i
                    in
                    range(0, total_length, self.max_seq_len)]
                for k, t in concatenated_examples.items()
            }
            return result
        # datasets = concatenate_datasets(datasets)

        datasets = datasets["train"].train_test_split(test_size=8192)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            batch_size=1024,
            num_proc=self.num_proc,
            remove_columns=["text"],
            desc="Tokenize texts"
        )

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1024,
            num_proc=self.num_proc,
            desc=f"Grouping texts in chunks of {self.max_seq_len}",
        )

        tokenized_datasets["train"].set_format(columns=["input_ids", "attention_mask"])
        tokenized_datasets["test"].set_format(columns=["input_ids", "attention_mask"])

        # 3. Store to disk as arrow (token_ids)
        tokenized_datasets.save_to_disk(self.cache_path)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator
            )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator
            )
