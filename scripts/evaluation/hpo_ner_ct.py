import os
import argparse
import pytorch_lightning
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from ner_model import NERModel
from radiology.wrist_xray_ner_datamodule import NERDataModule
from german_n2c2.n2c2_datamodule import NERDataModule as n2c2NERDataModule
from ggponc.ggponc_datamodule import GGPONCNERDataModule
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

parser = argparse.ArgumentParser(description="Train a segmentation model.")

parser.add_argument("--model_name", type=str, required=True, help="model_name")

parser.add_argument("--num_classes", type=int, required=True, help="num_classes")

parser.add_argument("--filename", type=str, required=True, help="location of data")

parser.add_argument("--save_dir", type=str, required=True, help="where to save models")
parser.add_argument("--storage_mode", type=str, default="sqlite", help="sqlite or postgresql")

parser.add_argument("--tmp_modelname", type=str, required=True, help="plain modelname")

parser.add_argument("--task", type=str, required=True, help="plain modelname")


args = parser.parse_args()


N_EPOCHS = 10 if args.task == "ggponc" else 100
N_TRIALS = 20 if args.task == "ggponc" else 100

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def test(lr:float,
         batch_size:int,
         warmup_steps: int,
         ):

    model = NERModel(modelname=args.modelname, lr=lr, warmup_steps=warmup_steps, num_classes=args.num_classes)

    if args.task == "wrist_ct":
        datamodule = NERDataModule(train_filename=args.train_filename,
                                val_filename=args.val_filename,
                                test_filename=args.test_filename,
                                batch_size=batch_size,
                                num_workers=4,
                                tokenizer_name=args.modelname)
    elif args.task == "n2c2":
        datamodule = n2c2NERDataModule(
            filename=args.filename, batch_size=batch_size, num_workers=4, tokenizer_name=args.modelname
        )
    elif args.task == "ggponc" or args.task == "grascco":
        datamodule = GGPONCNERDataModule(
            folder_name=args.filename, batch_size=None, num_workers=0, tokenizer_name=args.modelname
        )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        precision=32,
        max_epochs=N_EPOCHS,
        enable_checkpointing=False,
        accumulate_grad_batches=1,
        logger=TensorBoardLogger(save_dir=args.save_dir),  # "ner_wrist_ct/optuna-trial"
        callbacks=[
            EarlyStopping(monitor="Val/Token_F1", mode="max", patience=5),
        ],
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    auroc = trainer.callback_metrics["Test/AUROC"].item() * 100
    macro_f1 = trainer.callback_metrics["Test/F1"].item() * 100
    token_f1 = trainer.callback_metrics["Test/Token_F1"].item() * 100
    precision = trainer.callback_metrics["Test/Precision"].item() * 100
    recall = trainer.callback_metrics["Test/Recall"].item() * 100
    token_precision = trainer.callback_metrics["Test/Token_Precision"].item() * 100
    token_recall = trainer.callback_metrics["Test/Token_Recall"].item() * 100
    token_auroc = trainer.callback_metrics["Test/Token_AUROC"].item() * 100

    print('Model & AUROC & AUROC tok & Macro F1 & Token F1 & Precision & Token Precision & Token Precision & Recall & Token Recall\\\\')
    print(f'{args.modelname} & {auroc:.2f} & & {token_auroc:.2f} & {macro_f1:.2f} & {token_f1:.2f} & {precision:.2f} & {token_precision:.2f} & {recall:.2f} & {token_recall:.2f}')

    print('Model & Class & F1  \\\\')
    for c in range(model.num_classes):
        token_f1 = trainer.callback_metrics[f"Test/Token_F1_{c}"].item() * 100
        print(f'{token_f1:.2f}')

    print('Model & Class & Token_Precision\\\\')
    for c in range(model.num_classes):
        token_precision = trainer.callback_metrics[f"Test/Token_Precision_{c}"].item() * 100
        print(f'{token_precision:.2f}')

    print('Model & Class & Token_Recall  \\\\')
    for c in range(model.num_classes):
        token_recall = trainer.callback_metrics[f"Test/Token_Recall_{c}"].item() * 100
        print(f'{token_recall:.2f}')

    print('Model & Class & Token_AUROC  \\\\')
    for c in range(model.num_classes):
        token_auroc = trainer.callback_metrics[f"Test/Token_AUROC_{c}"].item() * 100
        print(f'{token_auroc:.2f}')


def objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_steps = trial.suggest_int("warmup_steps", 0, 10_000)

    model = NERModel(modelname=args.modelname, lr=lr, warmup_steps=warmup_steps, num_classes=args.num_classes)

    if args.task == "wrist_ct":
        datamodule = NERDataModule(train_filename=args.train_filename,
                                val_filename=args.val_filename,
                                test_filename=args.test_filename,
                                batch_size=batch_size,
                                num_workers=4,
                                tokenizer_name=args.modelname)

    elif args.task == "n2c2":
        datamodule = n2c2NERDataModule(
            filename=args.filename, batch_size=batch_size, num_workers=4, tokenizer_name=args.modelname
        )
    elif args.task == "ggponc" or args.task == "grascco":
        datamodule = GGPONCNERDataModule(
            folder_name=args.filename, batch_size=batch_size, num_workers=0, tokenizer_name=args.modelname
        )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        precision=32,
        max_epochs=N_EPOCHS,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        logger=TensorBoardLogger(save_dir=args.save_dir),  # "ner_wrist_ct/optuna-trial"
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="Val/Token_F1"),
            EarlyStopping(monitor="Val/Token_F1", mode="max", patience=5),
        ],
    )

    trainer.fit(model, datamodule)
    return float(trainer.callback_metrics["Val/Token_F1"].item())


if __name__ == "__main__":
    pytorch_lightning.seed_everything(42)
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    if args.storage_mode == "postgresql":
        storage_name = "postgresql://optuna:H2DtpMzm4zPej47v@postgres-optuna-service:5432/optunaDatabase"
    else:
        storage_name = f"sqlite:///hpo_{args.task}_{args.tmp_modelname}.db"

    if args.test.lower() == "false":
        study = optuna.create_study(
            load_if_exists=True,
            direction="maximize",
            pruner=pruner,
            storage=storage_name,
            study_name=f"hpo_{args.task}_{args.modelname}",
        )

        study.optimize(
            objective, n_trials=N_TRIALS, callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE, TrialState.PRUNED,))]
        )

        print("Number of finished trials: ", len(study.trials))

        if not args.storage_mode == "postgresql":
            print("Best trial:")

            trial = study.best_trial
            loaded_study = optuna.load_study(storage=storage_name, study_name=f"hpo_{args.task}_{args.tmp_modelname}")

            with open(f"best_trial_{args.task}_{args.tmp_modelname}.txt", "w") as f:
                params = loaded_study.best_params
                f.write(f"val_F1: {loaded_study.best_value}\n")
                for key in params.keys():
                    f.write(f"{key}: {params[key]}\n")
                    f.write("\n\n")
    else:
        study = optuna.load_study(storage=storage_name, study_name=f"hpo_{args.task}_{args.tmp_modelname}")
        best_params = study.best_params
        lr = best_params["lr"]
        batch_size = best_params["batch_size"]
        warmup_steps = best_params["warmup_steps"]

        test(lr, batch_size, warmup_steps)
        sys.exit(os.EX_OK)