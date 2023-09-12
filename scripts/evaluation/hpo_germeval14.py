import argparse
import os

import flair
import optuna
import pytorch_lightning
import torch

# dataset, model and embedding imports
from flair.datasets import NER_GERMAN_GERMEVAL
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from optuna._callbacks import MaxTrialsCallback
from optuna.trial import TrialState

parser = argparse.ArgumentParser(description="Train a segmentation model.")

parser.add_argument("--model_name", type=str, default="bert-base-german-cased", help="model_name")

parser.add_argument("--num_classes", type=int, required=True, help="num_classes")

parser.add_argument("--save_dir", type=str, required=True, help="where to save models")

parser.add_argument("--task", type=str, default="germeval-14", help="chest_ct or chest_xray")
parser.add_argument("--storage_mode", type=str, default="sqlite", help="sqlite or postgresql")
args = parser.parse_args()


def germeval(germeval_version, model_name, lr, batch_size):
    if germeval_version == 14:
        # All arguments that can be passed
        # use cuda device as passed
        # flair.device = f'cuda:{str(args.cuda)}'

        # for each passed seed, do one experimental run
        flair.set_seed(42)

        # model
        hf_model = model_name

        # initialize embeddings
        embeddings = TransformerWordEmbeddings(
            model=model_name,
            layers="-1",
            subtoken_pooling="first",
            fine_tune=True,
            use_context=False,
            respect_document_boundaries=False,
            use_auth_token=os.getenv("AUTHTOKEN", False),
        )

        # select dataset depending on which language variable is passed
        corpus = NER_GERMAN_GERMEVAL()

        # make the dictionary of tags to predict
        # tag_dictionary = corpus.make_tag_dictionary('ner')
        tag_dictionary = corpus.make_label_dictionary(label_type="ner")

        # init bare-bones sequence tagger (no reprojection, LSTM or CRF)
        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type="ner",
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )

        # init the model trainer
        trainer = ModelTrainer(tagger, corpus)

        # make string for output folder
        output_folder = f"flert-ner-{hf_model}-{42}"

        # train with XLM parameters (AdamW, 20 epochs, small LR)

        """
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            if self.warmup_steps == 0 and self.decay_steps == 0:
                return optimizer
            else:
                scheduler = transformers.get_constant_schedule_with_warmup(optimizer, self.warmup_steps)
                return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        """

        output = trainer.train(
            output_folder,
            learning_rate=lr,
            mini_batch_size=batch_size,
            max_epochs=30,
            optimizer=torch.optim.AdamW,
            embeddings_storage_mode="none",
            weight_decay=0.0,
            train_with_dev=False,
            num_workers=16,
        )
        return output["test_score"]


def objective(trial: optuna.trial.Trial) -> float:
    pytorch_lightning.seed_everything(42)

    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    score = germeval(germeval_version=14, model_name=args.model_name, lr=lr, batch_size=batch_size)

    return score


if __name__ == "__main__":
    pytorch_lightning.seed_everything(42)
    print("Initialize HPO Training")
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    if args.storage_mode == "postgresql":
        storage_name = "postgresql://optuna:H2DtpMzm4zPej47v@postgres-optuna-service:5432/optunaDatabase"
    else:
        storage_name = f"sqlite:///ner-wrist-ct-{args.tmp_modelname}.db"

    study = optuna.create_study(
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
        storage=storage_name,
        study_name=f"hpo-{args.task}_{args.model_name}",
    )

    study.optimize(
        objective, n_trials=100, callbacks=[MaxTrialsCallback(100, states=(TrialState.COMPLETE, TrialState.PRUNED,))]
    )

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")

    trial = study.best_trial

    loaded_study = optuna.load_study(storage=storage_name, study_name=f"hpo-{args.task}_{args.model_name}")

    with open("best_trial.txt", "w") as f:
        params = loaded_study.best_params
        f.write(f"val_F1: {loaded_study.best_value}\n")
        for key in params.keys():
            f.write(f"{key}: {params[key]}\n")
            f.write("\n\n")
