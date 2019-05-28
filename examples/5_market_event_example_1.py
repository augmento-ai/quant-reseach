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

# define the location of the input file
filename_augmento_topics = "data/example_data/augmento_topics.msgpack.zlib"
filename_augmento_data = "data/example_data/augmento_data.msgpack.zlib"
filename_bitmex_data = "data/example_data/bitmex_data.msgpack.zlib"

# load the example data
all_data = eh.load_example_data(filename_augmento_topics,
                             filename_augmento_data,
                             filename_bitmex_data,
                             datetime_start=datetime.datetime(2018, 8, 29),
                             datetime_end=datetime.datetime(2018, 9, 10))
aug_topics, aug_topics_inv, t_aug_data, aug_data, t_price_data, price_data = all_data

print(aug_topics_inv.keys())

# get the signals we're interested in
#aug_signal_a = aug_data[:, aug_topics_inv["Bullish"]].astype(np.float64)
#aug_signal_b = aug_data[:, aug_topics_inv["Bearish"]].astype(np.float64)
#aug_signal_a = aug_data[:, aug_topics_inv["Fork"]].astype(np.float64)


aug_data_smooth = np.array(aug_data, np.float64)
for i_sig in range(aug_data_smooth.shape[1]):
	aug_data_smooth[:, i_sig] = ah.causal_rolling_average(aug_data[:, i_sig], 24)
	aug_data_smooth[:, i_sig] = (aug_data_smooth[:, i_sig] - np.min(aug_data_smooth[:, i_sig])) / (np.max(aug_data_smooth[:, i_sig]) - np.min(aug_data_smooth[:, i_sig]))

#aug_signal_a = aug_data_smooth[:, aug_topics_inv["Bullish"]]
#aug_signal_a = aug_data_smooth[:, aug_topics_inv["Bearish"]]
aug_signal_a = aug_data_smooth[:, aug_topics_inv["Institutional_money"]]

# set up the figure
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)

# initialise some labels for the plot
datenum_aug_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_aug_data]
datenum_price_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_price_data]

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
ax[0].plot(datenum_price_data, price_data, linewidth=0.5)
#ax[1].plot(datenum_aug_data, aug_data.astype(np.float64), linewidth=0.5, alpha=0.5)
ax[1].plot(datenum_aug_data, aug_data_smooth, linewidth=0.5, alpha=0.5)
ax[1].plot(datenum_aug_data, aug_data_smooth[:, aug_topics_inv["Institutional_money"]], linewidth=0.5, color="b")
ax[1].plot(datenum_aug_data, aug_data_smooth[:, aug_topics_inv["Fearful/Concerned"]], linewidth=0.5, color="r")
#ax[1].plot(datenum_aug_data, aug_signal_b, linewidth=0.5, color="r")

# label axes
#ax[0].set_ylabel("__")
#ax[1].set_ylabel("__")
#ax[0].set_title("4_basic_strategy_example.py")

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d')
ax[0].xaxis.set_major_formatter(xfmt)

# show the plot
plt.show()
