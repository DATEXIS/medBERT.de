from typing import Any, List, Optional, Union

import apex
import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer


class MLMModel(pl.LightningModule):
    def __init__(
        self,
        num_hidden_layer: int = 12,
        num_attention_heads: int = 8,
        hidden_dim: int = 768,
        hf_checkpoint: str = None,
        tokenizer_name: str = "bert-base-uncased",
        warmup_steps: int = 500,
        decay_steps: int = 1_000_000,
        lr: float = 1e-4,
        weight_decay = 0.01, 
        optimizer: str = "not_lamb",
        huggingface_save_dir: str = "medbert",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        if hf_checkpoint is not None:
            self.model = BertForMaskedLM.from_pretrained(hf_checkpoint)
        else:
            config = BertConfig(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=hidden_dim,
                num_hidden_layers=num_hidden_layer,
                num_attention_heads=num_attention_heads,
                intermediate_size=4 * hidden_dim,
            )
            self.model = BertForMaskedLM(config)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.huggingface_save_dir = huggingface_save_dir
        self.weight_decay = weight_decay

    def training_step(self, batch, *args) -> STEP_OUTPUT:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels[input_ids != 4] = -100
        loss = self.model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)["loss"]
        self.log("Train/Loss", loss)
        return loss

    def validation_step(self, batch, *args) -> Optional[STEP_OUTPUT]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels[input_ids != 4] = -100
        loss = self.model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)["loss"]
        self.log("Val/Loss", loss)
        return loss

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if self.global_rank == 0:
            self.model.save_pretrained(self.huggingface_save_dir)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        weight_decay = 0.01
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if self.optimizer == "lamb":
            optimizer = apex.optimizers.FusedLAMB(optimizer_grouped_parameters, lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=weight_decay)

        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer, self.warmup_steps, self.decay_steps
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return [optimizer], [scheduler]
