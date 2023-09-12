import argparse
from pathlib import Path

import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Detect names in report texts.")

parser.add_argument(
    "--debug", action="store_true", default=False, help="Run with only a small fraction of the data for testing"
)

parser.add_argument("--debug_size", type=int, default=1e3, help="Reduced file size for debugging")

parser.add_argument(
    "--data_dir", type=str, default="/media/ScaleOut/database/text/", help="Root directory for data storage"
)

parser.add_argument(
    "--overwrite", action="store_true", default=False, help="Start new and overwrite the existing `names.csv`"
)


args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

reports = pd.read_csv(DATA_DIR / "tbase" / "Untersuchung_clean.csv", nrows=args.debug_size if args.debug else None)

if not Path("names_tbase.csv").exists() or args.overwrite:
    with open("names_tbase.csv", "w+") as f:
        f.write("UntersuchungID;name\n")
else:
    print("Continuing from existing csv")
    names = pd.read_csv("names_tbase.csv", low_memory=False, sep=";")
    names["UntersuchungID"] = names.UntersuchungID.astype(str)
    reports["UntersuchungID"] = reports.UntersuchungID.astype(str)
    print(len(reports))
    reports = reports[~reports.UntersuchungID.isin(names.UntersuchungID)]
    print(len(reports))

tagger = SequenceTagger.load("flair/ner-german-large")

with open("names_tbase.csv", "a+") as f:
    for idx, row in tqdm(reports.iterrows(), total=len(reports)):
        sentence = Sentence(row.text)
        tagger.predict(sentence)
        for label in sentence.get_labels():
            if label.value == "PER":
                name = label.shortstring.replace(label.value, "").replace('"', "").replace("/", "")
                line = f"{row.UntersuchungID};{name}\n"
                f.write(line)
