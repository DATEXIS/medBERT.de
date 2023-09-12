# 1. For each line in the dataset, generate a sparse bow vector
# 2. Batchwise Pairwise distance calculation
# 3. Grouping and reduction
#       - While there are items that can be grouped together (two very similar documents)
#       - Group always two documents together and consider, only one of them remains in the next step
#       - do this until there are no pairs anymore (distance between two documents > threshold)
# 4. Remaining documents are the dataset
from typing import List

import numpy as np
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer


def read_data():
    with open('./corpus_short_125_no_dup_with_type.csv', "r", encoding="utf-16") as reader:
        dataset = reader.readlines()
    
    #dataset = load_dataset('./corpus_short_125_no_dup_with_type.csv')
    texts = [l for l in dataset if l.strip() != ""][0:10000]
    # remove empty lines

    return texts


def group_similar_documents(distance_matrix: np.array, distance_threshold: float = 20) -> List[int]:
    """
    Deduplication based on a given pairwise distance matrix and a distance threshold.
    Returns a list of ids that can be used to reduce the dataset.
    :param distance_matrix:
    :param distance_threshold:
    :return:
    """
    distance_matrix[distance_matrix < distance_threshold] = 0

    distance_matrix = np.tri(*distance_matrix.shape) * distance_matrix
    distance_matrix[np.tri(*distance_matrix.shape, k=-1).T.astype(np.bool)] = -1

    documents_to_sample_from = np.logical_and(distance_matrix >= 0, distance_matrix < distance_threshold)
    index = np.arange(0,len(documents_to_sample_from))
    doc_ids = []
    for i in list(range(len(documents_to_sample_from)))[::-1]:
        if documents_to_sample_from[i].sum() > 0:
            relevant_ids = index[documents_to_sample_from[i]]
            doc_idx = np.random.choice(relevant_ids)
            documents_to_sample_from[:,doc_idx] = False
            doc_ids.append(doc_idx)

    return doc_ids

if __name__ == "__main__":
    # 1. Load data and remove empty lines (for testing we use wikitext)
    lines = read_data()

    # 2. Transform lines into bag of word vectors
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(lines)

    # 3. Calculate pairwise distances between all documents
    distances = sklearn.metrics.pairwise_distances(bow, bow)

    # 4. Now group until grouping doesn't reduce the matrix size anymore
    m = group_similar_documents(distances)
    print("Completed")