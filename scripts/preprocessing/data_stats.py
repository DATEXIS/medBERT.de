import re

import pandas as pd

# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from tqdm.notebook import tqdm


def number_of_words(x: str) -> int:
    if isinstance(x, str):
        return len(re.findall("[\w-]+", str(x)))  # noqa W605
    return 0


def number_of_sentences(x: str) -> int:
    if isinstance(x, str):
        return len(sent_tokenize(x))
    return 0


def dataframe_statistics(df: pd.DataFrame) -> dict:
    return {
        "n_words": [sum(df["n_words"])],
        "n_sentences": [sum(df["n_sentences"])],
        "n_documents": [len(df)],
    }


files = [
    "/media/ScaleOut/database/text/doccheck_flexikon/doccheck.csv",
    "/media/ScaleOut/database/text/ggponc/ggponc.csv",
    "/media/ScaleOut/database/text/webcrawl/webcrawl.csv",
    "/media/ScaleOut/database/text/pubmed/pubmed.csv",
    "/media/ScaleOut/database/text/radiology/radiology_anonym.csv",
    "/media/ScaleOut/database/text/springer/springer.csv",
    "/media/ScaleOut/database/text/tbase/tbase_anonym.csv",
    "/media/ScaleOut/database/text/thesis/thesis.csv",
    "/media/ScaleOut/database/text/thieme/thieme.csv",
    "/media/ScaleOut/database/text/wikipedia/wikipedia.csv",
    "/media/ScaleOut/database/text/wmt22/wmt22.csv",
]


statistics = []
for f in tqdm(files):
    dataframe = pd.read_csv(f)
    name = f.split(".")[-2].split("/")[-1]
    dataframe["n_words"] = [
        number_of_words(x) for x in tqdm(dataframe.text, postfix=f"{name} - counting words", leave=False)
    ]
    dataframe["n_sentences"] = [
        number_of_sentences(x) for x in tqdm(dataframe.text, postfix=f"{name} - counting sentences", leave=False)
    ]
    stats = dataframe_statistics(dataframe)
    statistics.append(stats)


statistics = pd.concat([pd.DataFrame(s) for s in statistics])

statistics["Name"] = [
    "DocCheck",
    "GGPONC",
    "Webcrawl",
    "Pubmed",
    "Radiology",
    "Spinger OA",
    "EHR",
    "Doctoral theses",
    "Thieme",
    "Wiki",
    "WMT22",
]

statistics = statistics[["Name", "n_documents", "n_sentences", "n_words"]]

summary = pd.DataFrame(
    {
        "Name": ["Summary"],
        "n_documents": [sum(statistics.n_documents)],
        "n_sentences": [sum(statistics.n_sentences)],
        "n_words": [sum(statistics.n_words)],
    }
)


print(pd.concat([statistics, summary]).to_markdown(index=False))
