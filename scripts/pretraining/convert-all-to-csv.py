import pandas as pd
from tqdm import tqdm

files = [
    "/media/ScaleOut/database/text/doccheck_flexikon/doccheck.csv",
    "/media/ScaleOut/database/text/ggponc/ggponc.csv",
    "/media/ScaleOut/database/text/webcrawl/webcrawl.csv",
    "/media/ScaleOut/database/text/pubmed/pubmed.csv",
    "/media/ScaleOut/database/text/radiology/radiology_anonym_no_duplicates.csv",
    "/media/ScaleOut/database/text/springer/springer.csv",
    "/media/ScaleOut/database/text/tbase/tbase_anonym.csv",
    "/media/ScaleOut/database/text/thesis/thesis.csv",
    "/media/ScaleOut/database/text/thieme/thieme.csv",
    "/media/ScaleOut/database/text/wikipedia/wikipedia.csv",
    "/media/ScaleOut/database/text/wmt22/wmt22.csv",
]

print("Reading data")
dataframes = [pd.read_csv(f)[["text"]] for f in tqdm(files)]
dataframes = pd.concat(dataframes)

print("Writing dataframe")
dataframes.to_csv("../../datasets/mlm_pretraining_data_no_duplicates.csv", index=False)

# print("Writing textdump for tokenizer")
# with open("../../datasets/tokenizer_data.txt", "w+") as f:
#     for text in dataframes.text:
#         if isinstance(text, str):
#             f.write(text)
#             f.write("\n")

