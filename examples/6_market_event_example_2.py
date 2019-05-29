import sys
import msgpack
import zlib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md

# import files from src
sys.path.insert(0, "src")
import example_helper as eh
import analysis_helper as ah
import data_loader_helper as dlh

datetime_start = datetime.datetime(2018, 10, 8)
datetime_end = datetime.datetime(2018, 12, 3, 1, 0, 0)

dlh.load_data(path_data="data/cache",
              augmento_coin="bitcoin",
              augmento_source="twitter",
              binance_symbol="BTCUSDT",
              dt_bin_size=3600,
              datetime_start=datetime_start,
              datetime_end=datetime_end,
              augmento_api_key=None)


quit()


# define the location of the input file
filename_augmento_topics = "data/example_data/augmento_topics.msgpack.zlib"
filename_augmento_data = "data/example_data/augmento_data.msgpack.zlib"
filename_bitmex_data = "data/example_data/bitmex_data.msgpack.zlib"

# load the example data
datetime_start = datetime.datetime(2018, 10, 8)
datetime_end = datetime.datetime(2018, 12, 3)
all_data = eh.load_example_data(filename_augmento_topics,
                             filename_augmento_data,
                             filename_bitmex_data,
                             datetime_start=datetime_start,
                             datetime_end=datetime_end)
aug_topics, aug_topics_inv, t_aug_data, aug_data, t_price_data, price_data = all_data

# initialise the smooth data
aug_data_smooth = aug_data.astype(np.float64)

# find the ratio of each signal to all signals at each time step
for i in range(aug_data_smooth.shape[0]):
	aug_data_smooth[i, :] = ah.safe_divide(aug_data_smooth[i, :], np.sum(aug_data_smooth[i, :]))

# smooth each signal
w = 24 * 7
for i_sig in range(aug_data_smooth.shape[1]):
	aug_data_smooth[:, i_sig] = ah.causal_rolling_average(aug_data_smooth[:, i_sig], w)

# remove the warm-up period for the smoothing function
aug_data_smooth[0:w, :] = np.ones((w, aug_data_smooth.shape[1]), dtype=np.float64) * aug_data_smooth[w, :]

# normalise each signal
for i_sig in range(aug_data_smooth.shape[1]):
	aug_data_smooth[:, i_sig] = (aug_data_smooth[:, i_sig] - np.min(aug_data_smooth[:, i_sig])) / (np.max(aug_data_smooth[:, i_sig]) - np.min(aug_data_smooth[:, i_sig]))

# set up the figure
fig, ax = plt.subplots(1, 1, sharex=True, sharey=False)

# initialise some labels for the plot
datenum_aug_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_aug_data]
datenum_price_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_price_data]

# plot stuff
ax.grid(linewidth=0.4)
ax.plot(datenum_price_data, price_data, linewidth=0.5,)
ax2 = ax.twinx()
ax2.plot(datenum_aug_data, aug_data_smooth[:, aug_topics_inv["Fork"]], linewidth=0.5, color="r", alpha=0.5)

# label axes
ax.set_ylabel("Price (XBt)")
ax2.set_ylabel("Fork")

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)

# show the plot
plt.show()
