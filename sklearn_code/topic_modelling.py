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

vect = TfidfVectorizer(stop_words = "english")
vect.fit(text)

vectorized_text = vect.transform(text)

words = vect.get_feature_names()

n_rows, n_columns = vectorized_text.shape

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
