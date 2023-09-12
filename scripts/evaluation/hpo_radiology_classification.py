import argparse
import os
import sys

import optuna
import pytorch_lightning
from optuna._callbacks import MaxTrialsCallback
from optuna.trial import TrialState
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from callbacks import PyTorchLightningPruningCallback
from classification_model import ClassificationModel
from germeval18_datamodule import Germeval18Datamodule
from medical_sat.medical_sat_datamodule import MedicalSatDataModule
from radiology.chest_ct_datamodule import CTDataModule
from radiology.chest_xray_datamodule import XRayDataModule
from social_media.lifeline_ADE_datamodule import LifelineDataModule

parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--model_name", type=str, required=True, help="model_name")
parser.add_argument("--num_classes", type=int, required=True, help="num_classes")
parser.add_argument("--train_filename", type=str, help="location of data")
parser.add_argument("--test_filename", type=str, help="location of data")
parser.add_argument("--val_filename", type=str, help="location of data")
parser.add_argument("--save_dir", type=str, required=True, help="where to save models")
parser.add_argument("--task", type=str, required=True, help="chest_ct, chest_xray or germeval18")
parser.add_argument("--task_type", type=str, required=True, help="chest_ct, chest_xray or germeval18")
parser.add_argument("--storage_mode", type=str, default="sqlite", help="sqlite or postgresql")
parser.add_argument("--tmp_modelname", type=str, help="modelname")
parser.add_argument("--test", type=str,  help="train the best model and log all metrics (HPO needs to be finished)")

args = parser.parse_args()



def init_datamodule(args, batch_size):
    if args.task == "chest_ct":
        datamodule = CTDataModule(
            train_filename=args.train_filename,
            test_filename=args.test_filename,
            val_filename=args.val_filename,
            batch_size=batch_size,
            num_workers=4,
            tokenizer_name=args.model_name,
        )

    elif args.task == "chest_xray":
        datamodule = XRayDataModule(
            train_filename=args.train_filename,
            test_filename=args.test_filename,
            val_filename=args.val_filename,
            batch_size=batch_size,
            num_workers=4,
            tokenizer_name=args.model_name,
        )

    elif args.task == "lifeline":
        datamodule = LifelineDataModule(
            filename=args.filename, batch_size=batch_size, num_workers=4, tokenizer_name=args.model_name
        )
    elif args.task == "medical_sat":
        datamodule = MedicalSatDataModule(
                            train_filename="/pvc/medical_sat_data/train.csv",
                            test_filename="",
                            val_filename = "",
                            batch_size=8,
                            num_workers=0,
                            tokenizer_name= args.model_name,
                            task_type='multiple_choice')
    else:
        datamodule = Germeval18Datamodule(batch_size=batch_size, num_workers=4, tokenizer_name=args.model_name)
    return datamodule


def objective(trial: optuna.trial.Trial) -> float:
    pytorch_lightning.seed_everything(42)

    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 64, 128])
    warmup_steps = trial.suggest_int("warmup_steps", 0, 10_000)
    accumulation_steps = trial.suggest_categorical("accumulation_steps", [1])
    model = ClassificationModel(
        modelname=args.model_name, lr=lr, warmup_steps=warmup_steps, num_classes=args.num_classes, task_type=args.task_type
    )

    datamodule = init_datamodule(args, batch_size)
    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=100,
        enable_checkpointing=False,
        accumulate_grad_batches=accumulation_steps,
        logger=TensorBoardLogger(save_dir=args.save_dir),  # "ner_wrist_ct/optuna-trial"
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="Val/Accuracy_Macro"),
            EarlyStopping(monitor="Val/Accuracy_Macro", mode="max", patience=5),
        ],
    )

    trainer.fit(model, datamodule)
    datamodule = init_datamodule(args, batch_size)
    trainer.test(model, datamodule)
    return float(trainer.callback_metrics["Test/Accuracy_Macro"].item())


def run_test_with_params(args, params) -> float:
    pytorch_lightning.seed_everything(42)

    lr = params["lr"]
    batch_size = params["batch_size"]
    warmup_steps = params["warmup_steps"]
    accumulation_steps = params["accumulation_steps"]
    batch_size = batch_size * accumulation_steps

    model = ClassificationModel(
        modelname=args.model_name, lr=lr, warmup_steps=warmup_steps, num_classes=args.num_classes
    )

    datamodule = init_datamodule(args, batch_size)
    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=100,
        accumulate_grad_batches=accumulation_steps,
        enable_checkpointing=False,
        logger=TensorBoardLogger(save_dir=args.save_dir),  # "ner_wrist_ct/optuna-trial"
        callbacks=[EarlyStopping(monitor="Test/Accuracy_Macro", mode="max", patience=5),],
    )

    trainer.fit(model, datamodule)
    datamodule = init_datamodule(args, batch_size)
    trainer.test(model, datamodule)

    auroc = trainer.callback_metrics["Test/AUROC"].item() * 100
    macro_f1 = trainer.callback_metrics["Test/F1_Macro"].item() * 100
    micro_f1 = trainer.callback_metrics["Test/F1_Micro"].item() * 100
    precision = trainer.callback_metrics["Test/Precision"].item() * 100
    recall = trainer.callback_metrics["Test/Recall"].item() * 100
    accuracy_macro = trainer.callback_metrics["Test/Accuracy_Macro"].item() * 100
    accuracy_micro = trainer.callback_metrics["Test/Accuracy_Micro"].item() * 100
    
    print('Model & AUROC & Macro F1 & Micro F1 & Precision & Recall & Macro Accuracy & Micro Accuracy \\\\')
    print(f'{args.modelname} & {auroc:.2f} & {macro_f1:.2f} & {micro_f1:.2f} & {precision:.2f} & {recall:.2f} & {accuracy_macro:.2f} & {accuracy_micro:.2f}')

    
    print('Model & Class & AUROC  \\\\')
    for c in range(model.num_classes):
        auroc = trainer.callback_metrics[f"Test/AUROC_{c}"].item() * 100
        print(f'{args.model_name} & {c} & {auroc:.2f}')

    print('Model & Class & F1  \\\\')
    for c in range(model.num_classes):
        f1 = trainer.callback_metrics[f"Test/F1_Macro_{c}"].item() * 100
        print(f'{args.model_name} & {c} & {f1:.2f}')

    print('Model & Class & Precision \\\\')
    for c in range(model.num_classes):
        precision = trainer.callback_metrics[f"Test/Precision_{c}"].item() * 100
        print(f'{args.model_name} & {c} & {precision:.2f}')
    
    print('Model & Class & Recall \\\\')
    for c in range(model.num_classes):
        recall = trainer.callback_metrics[f"Test/Recall_{c}"].item() * 100
        print(f'{args.model_name} & {c} & {recall:.2f}')
    
    print('Model & Class & Accuracy \\\\')
    for c in range(model.num_classes):
        accuracy = trainer.callback_metrics[f"Test/Accuracy_Macro_{c}"].item() * 100
        print(f'{args.model_name} & {c} & {accuracy:.2f}')






if __name__ == "__main__":
    pytorch_lightning.seed_everything(42)

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    if args.test.lower() == "false":
        if args.storage_mode == "postgresql":
            storage_name = "postgresql://optuna:H2DtpMzm4zPej47v@postgres-optuna-service.pgrundmann.svc.cluster.local:5432/optunaDatabase"
            study = optuna.create_study(
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
                storage=storage_name,
                study_name=f"hpo_{args.task}_{args.tmp_modelname}",
            )

        else:
            storage_name = f"sqlite:///hpo_{args.task}_{args.tmp_modelname}.db"

            study = optuna.create_study(
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
                storage=storage_name,
                study_name=f"hpo_{args.task}_{args.tmp_modelname}",
            )

        study.optimize(
            objective, n_trials=1000, callbacks=[MaxTrialsCallback(1000, states=(TrialState.COMPLETE, TrialState.PRUNED,))]
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
        if args.storage_mode == "postgresql":
            storage_name = "postgresql://optuna:H2DtpMzm4zPej47v@postgres-optuna-service.pgrundmann.svc.cluster.local:5432/optunaDatabase"
        else:
            storage_name = f"sqlite:///hpo_{args.task}_{args.tmp_modelname}.db"
        study = optuna.load_study(storage=storage_name, study_name=f"hpo_{args.task}_{args.tmp_modelname}")
        best_params = study.best_params

        run_test_with_params(args, best_params)
        sys.exit(os.EX_OK)