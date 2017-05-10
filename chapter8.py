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

stockdata_new = pd.DataFrame(stockdata, columns = ["stock", "open", "high", "low", "close", "volume"])
stockdata_new.head()


stockdata["previous_week_volume"] = 0
stockdata.head()
stockdata.dropna().head(2)

import numpy
stockdata_new.open.describe()


stockdata_new.open = pd.to_numeric(stockdata_new.open.str.replace("$", ""))
stockdata_new.close = pd.to_numeric(stockdata_new.close.str.replace("$", ""))
stockdata_new
pd.to_numeric(stockdata_new.close - stockdata_new.open)
stockdata_new.open.describe()

stockdata_new["newopen"] = stockdata_new.open.apply(lambda x: 0.8 * x)
stockdata_new.newopen.head(5)


stockAA = stockdata_new.query('stock=="AA"')
stockAA.head()



import matplotlib
import matplotlib.pyplot as plt
import numpy

stockCSCO = stockdata_new.query('stock == "CSCO"')
stockCSCO.head()

from matplotlib import figure
plt.figure()
plt.scatter(stockdata_new.index.date, stockdata_new.volume)
plt.xlabel("day")
plt.ylabel("stock close value")
plt.title("title")
plt.show()


plt.subplot(2, 2, 1)
plt.plot(stockAA.index.weekofyear, stockAA.open, "r--")
plt.subplot(2, 2, 2)
plt.plot(stockCSCO.index.weekofyear, stockCSCO.open, "g-*")
plt.subplot(2, 2, 3)
plt.plot(stockAA.index.weekofyear, stockAA.open, "g--")
plt.subplot(2, 2, 4)
plt.plot(stockCSCO.index.weekofyear, stockCSCO.open, "r-*")
plt.subplot(2, 2, 3)
# plt.plot(x, y, "g--")
plt.subplot(2, 2, 4)
# plt.plot()
plt.show()



### MORE ELEGANT
fig, axes = plt.subplots(nrows = 1, ncols = 2)
for ax in axes:
    ax.plot(x, y, "r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("title")

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y, r)





fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(stockAA.index.weekofyear, stockAA.open, label = "AA")
ax.plot(stockCSCO.index.weekofyear, stockCSCO.open, label = "CSCO")
ax.set_xlabel("weekofyear")
ax.set_ylabel("stock value")
ax.set_title("Weekly change in stock price")
ax.legend(loc = 2)
plt.show()




### Scatter plot
import matplotlib.pyplot as plt
plt.scatter(stockAA.index.weekofyear, stockAA.open)
plt.show()
plt.close()

### BAR CHART
n = 12
X = np.arange(n)
Y1 = np.random.uniform(0.5, 1.0, n)
Y2 = np.random.uniform(0.5, 1.0, n)
plt.bar(X, +Y1, facecolor = "#9999ff", edgecolor = "white")
plt.bar(X, -Y2, facecolor = "#ff9999", edgecolor = "white")
plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride = 1, cmap = "hot")
plt.show()
