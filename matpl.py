import matplotlib
import matplotlib.pyplot as plt
import numpy


import pandas as pd
import numpy as np
import pandas


stockdata = pd.read_csv("/Users/ilyaperepelitsa/Downloads/dow_jones_index/dow_jones_index.data",
                        parse_dates = ["date"], index_col = ["date"], nrows = 100)
stockdata_new = pd.DataFrame(stockdata, columns = ["stock", "open", "high",
                                                    "low", "close", "volume"])
stockdata_new.head()
stockCSCO = stockdata_new.query('stock == "CSCO"')
stockCSCO

plt.figure()
plt.scatter(stockdata_new.index.date, stockdata_new.volume)
plt.xlabel("day")
plt.ylabel("stock close value")
plt.title("title")
plt.show()
plt.savefig("matplot1.jpg")
