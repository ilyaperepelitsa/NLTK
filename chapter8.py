import pandas as pd
import numpy as np


stockdata = pd.read_csv("/Users/ilyaperepelitsa/Downloads/dow_jones_index/dow_jones_index.data",
                     parse_dates = ["date"], index_col = ["date"], nrows = 100)

stockdata.head()
max(stockdata["percent_change_price"])
max(stockdata["volume"])
stockdata.index
stockdata.index.day
stockdata.index.month
stockdata.index.year
# stockdata.resample("volume").apply(np.sum)
stockdata.drop(["percent_change_volume_over_last_wk"], axis = 1)

stockdata_new = pd.DataFrame(stockdata)
