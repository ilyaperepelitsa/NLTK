import pandas as pd
import numpy as np
import pandas


stockdata = pd.read_csv("/Users/ilyaperepelitsa/Downloads/dow_jones_index/dow_jones_index.data",
                        parse_dates = ["date"], index_col = ["date"], nrows = 100)

stockdata.head()

max(stockdata["volume"])
max(stockdata["percent_change_price"])
stockdata.index
stockdata.index.day
stockdata.index.month
stockdata.index.year

stockdata.resample("M", how = np.sum)


stockdata.drop(["percent_change_volume_over_last_wk"], axis = 1)
stockdata_new = pd.DataFrame(stockdata, columns = ["stock", "open", "high",
                                                    "low", "close", "volume"])
stockdata_new.head()


stockdata_new["volume"] = 0
stockdata_new.head()

stockdata.head()
stockdata.dropna().head(2)
stockdata.head()


stockdata_new.open.describe()
stockdata_new.head()
stockdata_new.open = stockdata_new.open.replace("$", "")
stockdata_new.close = stockdata_new.close.replace("$", "")
stockdata_new.open.describe()
stockdata_new.close.describe()
stockdata_new.head()


stockdata_new["newopen"] = stockdata_new.open.apply(lambda x: 0.8 * x)
stockdata_new.head()

stockAA = stockdata_new.query('stock == "AXP"')
stockAA.head()
