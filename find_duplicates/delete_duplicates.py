import pickle

import pandas as pd
from tqdm import tqdm

filename = "distances.pkl"

infile = open(filename, "rb")
distances = pickle.load(infile)
infile.close

path = "distances/corpus.csv"


def load_frame(path_to_df=path, encoding="utf-16", filter_value=0):

    df = pd.read_csv(path_to_df, encoding=encoding, index_col="id")
    df.drop([df.columns[0]], inplace=True, axis=1)
    df.drop_duplicates(subset=["text"], inplace=True)

    exam_type_distribution = df.groupby(["exam_type"])["exam_type"].count()
    exam_type_distribution.sort_values(ascending=False)

    list_filtered = exam_type_distribution[exam_type_distribution > filter_value].index
    df_filtered = df[df["exam_type"].isin(list_filtered)]

    print(f"original dataframe shape: {df.shape}")
    print(f"df_filtered shape: {df_filtered.shape}")

    print(f'numbers of exam types before: {len(set(df["exam_type"]))}')
    print(f"types after filtering: {list_filtered}")
    print(f'number of exam types after filtering: {len(set(df_filtered["exam_type"]))}')

    return df_filtered, list_filtered


if __name__ == "__main__":
    df = load_frame()[0]

    print(f"starting shape: {df.shape}")
    items_to_delete = []

    for index, (i, j) in tqdm(enumerate(distances.items())):
        if i not in items_to_delete:
            items_to_delete.extend(j[0])

    items_to_delete = list(set(items_to_delete))
    df = df.drop(items_to_delete)

    print(f"after deduplication the df has a shape of: {df.shape}")
    print("")

    df.to_csv("deduplicated.csv")
    print("finished creating new deduplicated csv file")
    print("")

    indices_kept = df.index.to_list()
    filename = "indices_kept"
    outfile = open(filename, "wb")
    pickle.dump(indices_kept, outfile)
    outfile.close()

    filename2 = "deleted_indices"
    outfile2 = open(filename2, "wb")
    pickle.dump(items_to_delete, outfile2)
    outfile2.close()
    print(f"finished creating indices files. {filename} and {filename2}")
