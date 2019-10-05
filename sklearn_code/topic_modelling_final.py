import collections
import numpy as np
import matplotlib.pyplot as plt
import random


from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import os


pew = pd.read_csv("/Volumes/data_pew/text_data/stack_overflow_pandas/SO_pandas.csv")
text = pew['Markdown'].dropna()

vect = TfidfVectorizer(stop_words = "english",min_df = 0.0001)
vect.fit(text)

vectorized_text = vect.transform(text)



# words = list(vect.vocabulary_.keys())
words = vect.get_feature_names()
words

reduce_dim = TruncatedSVD(n_components = 1000)
reduce_dim.fit(vectorized_text)

# sum(reduce_dim.explained_variance_ratio_)
vectorized_reduced = reduce_dim.transform(vectorized_text)


# pd.DataFrame(vectorized_reduced)
list_kmeans = [MiniBatchKMeans(n_clusters = i, verbose = 0).fit(vectorized_reduced) for i in range(1, 100)]
# [i.inertia_ for i in list_kmeans]
clusters_data = pd.DataFrame([{"n_clusters" : i[0], "inertia" : i[1]} for i in zip(range(1, 100),[k.inertia_ for k in list_kmeans])])
plt.plot(clusters_data['n_clusters'],clusters_data['inertia'])


kmeans_tested = KMeans(n_clusters=30, n_jobs = -1, verbose = 2)
kmeans_tested.fit(vectorized_reduced)


# pd.DataFrame(kmeans_tested.predict(vectorized_reduced))

# pd.DataFrame(vectorized_reduced)


data_plot = pd.concat([pd.DataFrame(vectorized_reduced), pd.Series(kmeans_tested.predict(vectorized_reduced), name = "cluster")], axis = 1)
# pd.Series(kmeans_tested.predict(vectorized_reduced), name = "cluster")

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
for i in data_plot['cluster'].unique().tolist()[10:13]:
    plt.scatter(data_plot.loc[data_plot['cluster'] == i,0],
                data_plot.loc[data_plot['cluster'] == i,4], alpha = 0.1)



svd_components = pd.DataFrame(reduce_dim.components_.T, index = words)

#
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],0].sort_values(ascending = False)
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],0].sort_values(ascending = False)
#
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],0].sort_values(ascending = False)
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],0].sort_values(ascending = False)
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],1].sort_values(ascending = False)
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],1].sort_values(ascending = False)
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],2].sort_values(ascending = False)
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],2].sort_values(ascending = False)
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],3].sort_values(ascending = False)
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],3].sort_values(ascending = False)
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],4].sort_values(ascending = False)
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],4].sort_values(ascending = False)
# svd_components.loc[svd_components[0] > svd_components[0].describe(percentiles = [0.001, 0.999])["99.9%"],5].sort_values(ascending = False)
# svd_components.loc[svd_components[0] < svd_components[0].describe(percentiles = [0.001, 0.999])["0.1%"],5].sort_values(ascending = False)
for i in svd_components.columns.tolist()[0:5]:
    print("====================================================")
    print(svd_components.loc[svd_components[i] > svd_components[i].describe(percentiles = [0.001, 0.999])["99.9%"],i].sort_values(ascending = False))
    print(svd_components.loc[svd_components[i] < svd_components[i].describe(percentiles = [0.001, 0.999])["0.1%"],i].sort_values(ascending = False))




cluster_list = [random.choice(data_plot['cluster'].unique().tolist()) for i in range(0, 5)]
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
for cluster in cluster_list:
    # cluster = random.choice(data_plot['cluster'].unique().tolist())
    plt.scatter(data_plot.loc[data_plot['cluster'] == cluster,0],
                data_plot.loc[data_plot['cluster'] == cluster,1], alpha = 0.1)

    plt.scatter(data_plot.loc[data_plot['cluster'] == cluster,1],
                data_plot.loc[data_plot['cluster'] == cluster,2], alpha = 0.1)



# svd_components[0].describe(percentiles = [0.001, 0.999])
