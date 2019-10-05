import pandas as pd
import matplotlib.pyplot as plt


svd_data = pd.read_csv('svd_test.csv')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for i in sorted(svd_data["n_columns"].unique().tolist()):
    plt.plot(svd_data.loc[svd_data["n_columns"] == i,"components"],
            svd_data.loc[svd_data["n_columns"] == i,"explained_variance"],
            label='vocab size: {n_columns}'.format(**{"n_columns" : str(i)}))
    plt.legend(loc='best')
plt.show()
