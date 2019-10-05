import collections
import numpy as np
import matplotlib.pyplot as plt



from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import os


pew = pd.read_csv("/Volumes/data_pew/text_data/stack_overflow_pandas/SO_pandas.csv")
text = pew['Markdown'].dropna()

vect = TfidfVectorizer(stop_words = "english", min_df = 0.001)
vect.fit(text)

vectorized_text = vect.transform(text)
vectorized_text.shape


# words = list(vect.vocabulary_.keys())
words = vect.get_feature_names()
# words
# sorted(vect.vocabulary_.items(), key = lambda kv:(-kv[1]))
# vectorized_text.shape
# vectorized_df = pd.DataFrame(vectorized_text)
# sp_df = pd.SparseDataFrame(vectorized_text)
# kmeans = KMeans(n_clusters=2, n_jobs = -1, verbose = 2)
# kmeans.fit(vectorized_text)
#
#
# mini_kmeans_2 = MiniBatchKMeans(n_clusters=2, verbose = 1)
# mini_kmeans_2.fit(vectorized_text)
# mini_kmeans_2.inertia_
#
#
# mini_kmeans_3 = MiniBatchKMeans(n_clusters=3, verbose = 1)
# mini_kmeans_3.fit(vectorized_text)
# mini_kmeans_3.inertia_

# list_kmeans = [MiniBatchKMeans(n_clusters = i, verbose = 0).fit(vectorized_text) for i in range(1, 10)]
#
#
# reduction = TruncatedSVD(n_components = 500)
# reduction.fit(vectorized_text)
n_rows, n_columns = vectorized_text.shape
#
#
# sum(reduction.explained_variance_ratio_)
# reduction.explained_variance_

for i in range(0, 1000):
    try:
        check_combo = pd.read_csv('svd_test.csv')
        if check_combo.loc[((check_combo.components == i) &
                            (check_combo.n_rows == n_rows) &
                            (check_combo.n_columns == n_columns) &
                            (check_combo.experiment == "stack_overflow"))
                        ,:].shape[0] == 0:
            reduction = TruncatedSVD(n_components = i)
            reduction.fit(vectorized_text)
            file_exists = os.path.isfile("svd_test.csv")
            if file_exists:
                pd.DataFrame(pd.Series({"components": i,
                            "explained_variance": sum(reduction.explained_variance_ratio_),
                            "n_rows" : n_rows,
                            "n_columns" : n_columns,
                            "experiment" : "stack_overflow"})).T.loc[:,["components", "explained_variance", "n_rows", "n_columns", "experiment"]].\
                            to_csv('svd_test.csv', mode='a', header=False, index = False)
            else:
                pd.DataFrame(pd.Series({"components": i,
                            "explained_variance": sum(reduction.explained_variance_ratio_),
                            "n_rows" : n_rows,
                            "n_columns" : n_columns,
                            "experiment" : "stack_overflow"})).T.loc[:,["components", "explained_variance", "n_rows", "n_columns", "experiment"]].\
                            to_csv('svd_test.csv', mode='a', header=True, index = False)
        else:
            pass
    except:
        reduction = TruncatedSVD(n_components = i)
        reduction.fit(vectorized_text)
        file_exists = os.path.isfile("svd_test.csv")
        if file_exists:
            pd.DataFrame(pd.Series({"components": i,
                        "explained_variance": sum(reduction.explained_variance_ratio_),
                        "n_rows" : n_rows,
                        "n_columns" : n_columns,
                        "experiment" : "stack_overflow"})).T.loc[:,["components", "explained_variance", "n_rows", "n_columns", "experiment"]].\
                        to_csv('svd_test.csv', mode='a', header=False, index = False)
        else:
            pd.DataFrame(pd.Series({"components": i,
                        "explained_variance": sum(reduction.explained_variance_ratio_),
                        "n_rows" : n_rows,
                        "n_columns" : n_columns,
                        "experiment" : "stack_overflow"})).T.loc[:,["components", "explained_variance", "n_rows", "n_columns", "experiment"]].\
                        to_csv('svd_test.csv', mode='a', header=True, index = False)


# check_combo = pd.read_csv('svd_test.csv')

n_rows, n_columns

# check_combo.loc[((check_combo.components == 1) &
#                 (check_combo.components == 1) &
#                 (check_combo.components == 1) &
#                 (check_combo.components == 1))
#                 ,:]

# [i.inertia_ for i in list]


# pd.DataFrame(pd.Series({"components": i,"explained_variance": "pew"})).T
