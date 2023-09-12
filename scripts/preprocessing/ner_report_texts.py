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

reports = pd.read_csv(DATA_DIR / "radiology" / "report_texts_clean.csv", nrows=args.debug_size if args.debug else None)
reports["accessions_number"] = reports["accessions_number"].astype("string")

if not Path("names.csv").exists() or args.overwrite:
    with open("names.csv", "w+") as f:
        f.write("parent,filename,accessions_number,name\n")
else:
    names = pd.read_csv("names_radiology.csv")
    names["accessions_number"] = names["accessions_number"].astype("string")
    reports = reports[~reports.accessions_number.isin(names.accessions_number)]

tagger = SequenceTagger.load("flair/ner-german-large")

with open("names_radiology.csv", "a+") as f:
    for idx, row in tqdm(reports.iterrows(), total=len(reports)):
        sentence = Sentence(row.text)
        tagger.predict(sentence)
        for label in sentence.get_labels():
            if label.value == "PER":
                name = label.shortstring.replace(label.value, "").replace('"', "").replace("/", "")
                line = f"{row.parent},{row.filename},{row.accessions_number},{name}\n"
                f.write(line)
