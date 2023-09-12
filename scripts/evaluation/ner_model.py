import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics.classification
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import BertForTokenClassification, RobertaForTokenClassification, AutoModelForTokenClassification


class NERModel(pl.LightningModule):
    def __init__(
            self,
            num_classes: int = 58,
            modelname="bert-base-german-cased",
            lr=1e-5,
            warmup_steps: int = 0,
            decay_steps: int = 50_000,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.model = AutoModelForTokenClassification.from_pretrained(
            modelname, num_labels=num_classes, use_auth_token=os.getenv("AUTHTOKEN", False)
        )

        self.auroc = torchmetrics.classification.AUROC(num_classes=num_classes, task="multiclass", average="macro")
        self.f1 = torchmetrics.classification.F1Score(num_classes=num_classes, task="multiclass", average="macro")
        self.model_precision = torchmetrics.Precision(num_classes=num_classes, task="multiclass", average="macro")
        self.model_recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass", average="macro")

        # Token-based metrics
        self.tok_auroc = torchmetrics.classification.AUROC(num_classes=num_classes, task="multiclass", average="macro")
        self.tok_f1 = torchmetrics.classification.F1Score(num_classes=num_classes, task="multiclass", average="macro")
        self.tok_model_precision = torchmetrics.Precision(num_classes=num_classes, task="multiclass", average="macro")
        self.tok_model_recall = torchmetrics.Recall(num_classes=num_classes, task="multiclass", average="macro")

        self.tok_auroc_test = torchmetrics.classification.AUROC(num_classes=num_classes, task="multiclass",
                                                                average=None)
        self.tok_f1_test = torchmetrics.classification.F1Score(num_classes=num_classes, task="multiclass", average=None)
        self.tok_model_precision_test = torchmetrics.Precision(num_classes=num_classes, task="multiclass", average=None)
        self.tok_model_recall_test = torchmetrics.Recall(num_classes=num_classes, task="multiclass", average=None)

        self.save_hyperparameters()

    def training_step(self, batch, *args) -> STEP_OUTPUT:
        loss = self.model(
            input_ids=batch["input_ids"].reshape(-1, 512),
            attention_mask=batch["attention_mask"].reshape(-1, 512),
            labels=batch["labels"].reshape(-1, 512)
        ).loss
        self.log("Train/Loss", loss)
        return loss

    def test_step(self, batch, *args) -> Optional[STEP_OUTPUT]:
        out = self.model(input_ids=batch["input_ids"].reshape(-1, 512),
                         attention_mask=batch["attention_mask"].reshape(-1, 512),
                         labels=batch["labels"].reshape(-1, 512))
        logits = out.logits.reshape((-1, self.num_classes))
        logits = logits[batch["labels"].reshape((-1)) != -100]
        labels = batch["labels"][batch["labels"] != -100]
        self.f1(logits, labels)
        self.auroc(logits, labels)
        self.model_precision(logits, labels)
        self.model_recall(logits, labels)

        token_logits = out.logits.reshape((-1, self.num_classes))
        token_logits = token_logits[batch["word_mask"].reshape((-1)) != -100]
        token_labels = batch["labels"][batch["word_mask"] != -100]

        self.tok_f1(token_logits, token_labels)
        self.tok_auroc(token_logits, token_labels)
        self.tok_model_precision(token_logits, token_labels)
        self.tok_model_recall(token_logits, token_labels)

        self.tok_f1_test(token_logits, token_labels)
        self.tok_auroc_test(token_logits, token_labels)
        self.tok_model_precision_test(token_logits, token_labels)
        self.tok_model_recall_test(token_logits, token_labels)

        self.log("Test/Loss", out.loss)

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            {
                "Test/F1": self.f1.compute(),
                "Test/AUROC": self.auroc.compute(),
                "Test/Precision": self.model_precision.compute(),
                "Test/Recall": self.model_recall.compute(),
                "Test/Token_F1": self.tok_f1.compute(),
                "Test/Token_AUROC": self.tok_auroc.compute(),
                "Test/Token_Precision": self.tok_model_precision.compute(),
                "Test/Token_Recall": self.tok_model_recall.compute(),
            }
        )

        tok_f1 = self.tok_f1_test.compute()
        tok_auc = self.tok_auroc_test.compute()
        tok_model_precision = self.tok_model_precision_test.compute()
        tok_model_recall = self.tok_model_recall_test.compute()

        for c in range(self.num_classes):
            self.log_dict(
                {
                    f"Test/Token_F1_{c}": tok_f1[c],
                    f"Test/Token_AUROC_{c}": tok_auc[c],
                    f"Test/Token_Precision_{c}": tok_model_precision[c],
                    f"Test/Token_Recall_{c}": tok_model_recall[c],
                }
            )

    def validation_step(self, batch, *args) -> Optional[STEP_OUTPUT]:
        out = self.model(input_ids=batch["input_ids"].reshape(-1, 512),
                         attention_mask=batch["attention_mask"].reshape(-1, 512),
                         labels=batch["labels"].reshape(-1, 512))

        logits = out.logits.reshape((-1, self.num_classes))
        logits = logits[batch["labels"].reshape((-1)) != -100]
        labels = batch["labels"][batch["labels"] != -100]
        self.f1(logits, labels)
        self.auroc(logits, labels)
        self.model_precision(logits, labels)
        self.model_recall(logits, labels)

        token_logits = out.logits.reshape((-1, self.num_classes))
        token_logits = token_logits[batch["word_mask"].reshape((-1)) != -100]
        token_labels = batch["labels"][batch["word_mask"] != -100]

        self.tok_f1(token_logits, token_labels)
        self.tok_auroc(token_logits, token_labels)
        self.tok_model_precision(token_logits, token_labels)
        self.tok_model_recall(token_logits, token_labels)

        self.log("Val/Loss", out.loss)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                "Val/F1": self.f1.compute(),
                "Val/AUROC": self.auroc.compute(),
                "Val/Precision": self.model_precision.compute(),
                "Val/Recall": self.model_recall.compute(),
                "Val/Token_F1": self.tok_f1.compute(),
                "Val/Token_AUROC": self.tok_auroc.compute(),
                "Val/Token_Precision": self.tok_model_precision.compute(),
                "Val/Token_Recall": self.tok_model_recall.compute(),
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.warmup_steps == 0 and self.decay_steps == 0:
            return optimizer
        else:
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, self.warmup_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
