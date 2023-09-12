import os
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import AUROC
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from transformers import AutoModelForSequenceClassification, AutoModelForMultipleChoice


class ClassificationModel(pl.LightningModule):
    def __init__(
            self,
            num_classes: int = 24,
            modelname="bert-base-german-cased",
            lr=1e-5,
            warmup_steps: int = 0,
            decay_steps: int = 50_000,
            task_type: str = "multiple_choice"
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        if task_type == 'classification':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                modelname, num_labels=num_classes, use_auth_token=os.getenv("AUTHTOKEN", False),
            )
        elif task_type == 'multiple_choice':
            self.model = AutoModelForMultipleChoice.from_pretrained(
                modelname, num_labels=num_classes, use_auth_token=os.getenv("AUTHTOKEN", False),
            )
        else:
            raise Exception('give right task_type name')

        task = "multiclass"

        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass", average='macro')
        self.accuracy_micro = Accuracy(num_classes=num_classes, task="multiclass", average='micro')
        self.auroc = AUROC(num_classes=num_classes, task="multiclass", average="macro")
        self.precision_metric = Precision(num_classes=num_classes, task="multiclass", average="macro")
        self.recall = Recall(num_classes=num_classes, task="multiclass", average="macro")
        self.f1 = F1Score(num_classes=num_classes, task="multiclass", average="macro")
        self.f1_micro = F1Score(num_classes=num_classes, task="multiclass", average="micro")

        self.accuracy_test = Accuracy(num_classes=num_classes, task="multiclass", average=None)
        self.auroc_test = AUROC(num_classes=self.num_classes, task="multiclass", average=None)
        self.precision_metric_test = Precision(num_classes=self.num_classes, task="multiclass", average=None)
        self.recall_test = Recall(num_classes=self.num_classes, task="multiclass", average=None)
        self.f1_test = F1Score(num_classes=self.num_classes, task="multiclass", average=None)

        self.save_hyperparameters()

    def forward(self, batch, *args) -> STEP_OUTPUT:
        if self.num_classes == 2:
            batch["labels"] = batch["labels"].long()
        out = self.model(**batch)
        loss = out.loss
        self.log("Train/Loss", loss)
        return loss

    def training_step(self, batch, *args) -> STEP_OUTPUT:
        if self.num_classes == 2:
            batch["labels"] = batch["labels"].long()
        out = self.model(**batch)
        loss = out.loss
        self.log("Train/Loss", loss)
        return loss

    def validation_step(self, batch, *args) -> Optional[STEP_OUTPUT]:
        if self.num_classes == 2:
            batch["labels"] = batch["labels"].long()
        out = self.model(**batch)
        self.auroc(out.logits, batch["labels"].long())
        self.precision_metric(out.logits, batch["labels"].long())
        self.recall(out.logits, batch["labels"].long())
        self.f1(out.logits, batch["labels"].long())
        self.f1_micro(out.logits, batch["labels"].long())
        self.accuracy(out.logits, batch["labels"].long())
        self.accuracy_micro(out.logits, batch["labels"].long())
        self.log("Val/Loss", out.loss)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        auroc = self.auroc.compute()
        precision = self.precision_metric.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        f1_micro = self.f1_micro.compute()
        accuracy = self.accuracy.compute()
        accuracy_micro = self.accuracy_micro.compute()

        self.log("Val/Accuracy_Macro", accuracy)
        self.log("Val/Accuracy_Micro", accuracy_micro)
        self.log("Val/AUROC", auroc)
        self.log("Val/Precision", precision)
        self.log("Val/Recall", recall)
        self.log("Val/F1-Macro", f1)
        self.log("Val/F1-Micro", f1_micro)

    def test_step(self, batch, *args) -> Optional[STEP_OUTPUT]:
        if self.num_classes == 2:
            batch["labels"] = batch["labels"].long()
        out = self.model(**batch)
        self.auroc_test(out.logits, batch["labels"].long())
        self.precision_metric_test(out.logits, batch["labels"].long())
        self.recall_test(out.logits, batch["labels"].long())
        self.f1_test(out.logits, batch["labels"].long())
        self.accuracy_test(out.logits, batch["labels"].long())

        self.auroc(out.logits, batch["labels"].long())
        self.precision_metric(out.logits, batch["labels"].long())
        self.recall(out.logits, batch["labels"].long())
        self.f1(out.logits, batch["labels"].long())
        self.f1_micro(out.logits, batch["labels"].long())
        self.accuracy(out.logits, batch["labels"].long())
        self.accuracy_micro(out.logits, batch["labels"].long())

        self.log("Test/Loss", out.loss)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        auroc = self.auroc_test.compute()
        precision = self.precision_metric_test.compute()
        recall = self.recall_test.compute()
        f1 = self.f1_test.compute()
        accuracy = self.accuracy_test.compute()
        assert len(auroc) == self.num_classes

        for c in range(self.num_classes):
            self.log(f"Test/AUROC_{c}", auroc[c])
        for c in range(self.num_classes):
            self.log(f"Test/Precision_{c}", precision[c])
        for c in range(self.num_classes):
            self.log(f"Test/Recall_{c}", recall[c])
        for c in range(self.num_classes):
            self.log(f"Test/F1_Macro_{c}", f1[c])
        for c in range(self.num_classes):
            self.log(f"Test/Accuracy_Macro_{c}", accuracy[c])

        auroc = self.auroc.compute()
        precision = self.precision_metric.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        f1_micro = self.f1_micro.compute()
        accuracy = self.accuracy.compute()
        accuracy_micro = self.accuracy_micro.compute()

        self.log("Test/Accuracy_Macro", accuracy)
        self.log("Test/Accuracy_Micro", accuracy_micro)
        self.log("Test/AUROC", auroc)
        self.log("Test/Precision", precision)
        self.log("Test/Recall", recall)
        self.log("Test/F1_Macro", f1)
        self.log("Test/F1_Micro", f1_micro)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.warmup_steps == 0 and self.decay_steps == 0:
            return optimizer
        else:
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, self.warmup_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
