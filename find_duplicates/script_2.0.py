from time import time
from time import sleep
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer as skTfidf
import cupy as cp           #use cupy array instead of numpy to speed up calculation by using GPU
import cudf as cf
from cuml.metrics.pairwise_distances import sparse_pairwise_distances
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidf
from sklearn.metrics import pairwise_distances


import matplotlib.pyplot as plt
import sys

path = 'distances/corpus.csv'
# path = '/Users/lsacy/data/corpus.csv'
#path = './corpus.csv'
def load_frame(path_to_df=path, encoding='utf-16', filter_value = 100000):
    
    df= pd.read_csv(path_to_df, encoding=encoding, index_col='id')
    df.drop([df.columns[0]], inplace=True, axis=1)
    df.drop_duplicates(subset=['text'],inplace=True)
    
    exam_type_distribution = df.groupby(['exam_type'])['exam_type'].count()
    exam_type_distribution.sort_values(ascending=False)
    
    list_filtered = exam_type_distribution[exam_type_distribution > filter_value].index
    df_filtered = df[df['exam_type'].isin(list_filtered)]
    
    print(f'original dataframe shape: {df.shape}')
    print(f'df_filtered shape: {df_filtered.shape}')
    
    print(f'numbers of exam types before: {len(set(df["exam_type"]))}')
    print(f'types after filtering: {list_filtered}')
    print(f'number of exam types after filtering: {len(set(df_filtered["exam_type"]))}')

    return df_filtered, list_filtered

    # define a load function to load each exam type
def load_exam_type(df, exam_types):
    for i in exam_types:
        print(f'exam type: {i}')
        dataframe = df[df['exam_type'] == i]    
        print(f'number of documents: {dataframe.shape}')
        yield dataframe

def batch_tfidf(sparseMatrix, size = 5000):
    for idx, item in enumerate(range(0, sparseMatrix.shape[0], size)):
        batch_sparseMatrix = sparseMatrix[item:item+size,:]
        print(f'batch shape: {batch_sparseMatrix.shape}, item: {item} - {item+size}')
        yield batch_sparseMatrix

def get_distances(df_filtered, upperbound, lower, batch_size=8000, filter=100000):
    df_filtered = load_frame(filter_value = filter)
    df_by_type = load_exam_type(df_filtered[0], df_filtered[1])
    results_dict = {}

    # loop through each exam type
    for i in range(0,len(df_filtered[1])):
        print(f'exam {i+1}/{len(df_filtered[1])}')
        dataframe = next(df_by_type)
        df_indices = dataframe.index.to_list()

        tfidf = cuTfidf().fit_transform(dataframe['text'])
        
        batch = batch_tfidf(tfidf, size=batch_size)
        total_number_batches = math.ceil(tfidf.shape[0]/batch_size)
        print(f'# of batches: {total_number_batches} | batch size: {batch_size}')
        
        # loop through batches of tfidf matrix row wise
        counter = 0
        for i in range(0, total_number_batches):
            print(f'batch {i+1}/{total_number_batches}')
            
            batch_sparse = next(batch)
            
            distance_batch = sparse_pairwise_distances(batch_sparse, tfidf, metric='euclidean') # distance matrix

            sort_by_distance(distance_batch, df_indices, upperbound, results_dict, lower)
                                         
            del distance_batch
            del batch_sparse
                                         
            counter += batch_size
            print(f'results dict length: {len(results_dict)}')
            print('')
            


        del tfidf
        del batch
        sleep(5)
                
    return results_dict, df_filtered[0]

def sort_by_distance(distance_batch, df_indices, upperbound, results, lower):
    #distance_batch= distance_batch[0:100] #take a sample of 100
    found = 0
    for i, row in tqdm(enumerate(distance_batch)):
        sorted_array = cp.sort(row)
        arg_sorted = cp.argsort(row)

        candidates, distances = get_candidates(sorted_array, lower, arg_sorted, upperbound)


        if len(candidates) > 1:
            found += 1

            df_candidates = [df_indices[int(i)] for i in candidates]
            original_index = df_candidates[0]
            df_candidates = df_candidates[1:]
            
            save_results(distances, results, original_index, df_candidates)
            
            del sorted_array, original_index, df_candidates, distances, arg_sorted, candidates
            
    print(f'{found} matches found')

    return results

def save_results(distances, results, original_index, df_candidates):
    results[original_index] = (df_candidates, distances)
    return results

def get_candidates(sorted_array, lower, arg_sorted, upper):
    candidates = []
    distances = []
    for i, x in enumerate(sorted_array):
        if x > upper:
            break
        elif x < lower:
            continue
        else:
            candidates.append(arg_sorted[i])
            distances.append(float(x))
    return candidates, distances[1:]

import pickle


def main():
    results, df = get_distances(path, upperbound=0.75, lower=0, batch_size=3000, filter=10000)
    return results, df

if __name__ == "__main__":
    results, df = main()
    filename = 'distances.pkl'
    outfile = open(filename, 'wb')
    pickle.dump(results, outfile)
    outfile.close()