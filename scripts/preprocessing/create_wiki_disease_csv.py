import pandas as pd

fnames = [
    "wikisection_de_disease_articles",
    "wikisection_de_disease_test",
    "wikisection_de_disease_train",
    "wikisection_de_disease_validation",
]


for fname in fnames:
    path = f"/media/ScaleOut/database/text/wikisection_de_disease/{fname}.json"
    x = pd.read_json(path)
    x.to_csv(fname + ".csv", index=False)
