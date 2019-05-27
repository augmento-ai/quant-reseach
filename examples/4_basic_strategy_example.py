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
                             filename_bitmex_data)
aug_topics, aug_topics_inv, t_aug_data, aug_data, t_price_data, price_data = all_data

# strip the sentiments and prices outside the shared time range
t_start = max(np.min(t_aug_data), np.min(t_price_data))
t_end = min(np.max(t_aug_data), np.max(t_price_data))
t_aug_data, aug_data = ah.strip_data_by_time(t_aug_data, aug_data, t_start, t_end)
t_price_data, price_data = ah.strip_data_by_time(t_price_data, price_data, t_start, t_end)

# get the signals we're interested in
aug_signal_a = aug_data[:, aug_topics_inv["Bullish"]].astype(np.float64)
aug_signal_b = aug_data[:, aug_topics_inv["Bearish"]].astype(np.float64)

# generate a non-stationary ratio
n_days = 7
window_size = 24 * n_days
sent_ratio = ah.safe_divide(aug_signal_a, aug_signal_b)
sent_ratio_smooth = ah.causal_rolling_average(sent_ratio, window_size)
sent_score = ah.causal_rolling_sd(sent_ratio_smooth, 24*n_days)

# calculate the pnl (very basic backtest)
buy_sell_fee = 0.0
pnl = np.zeros(price_data.shape)
pnl[0] = 1.0
for i_p in range(price_data.shape[0])[1:]:
	
	# if sentiment score is positive, simulate long position
	# else if sentiment score is negative, simulate short position
	# (note that this is a very approximate market simulation!)
	if sent_score[i_p-1] > 0.0:
		pnl[i_p] = (price_data[i_p] / price_data[i_p-1]) * pnl[i_p-1]
	elif sent_score[i_p-1] <= 0.0:
		pnl[i_p] = (price_data[i_p-1] / price_data[i_p]) * pnl[i_p-1]
	
	# simulate a trade fee if we cross from long to short, or visa versa
	if np.sign(sent_score[i_p]) != np.sign(sent_score[i_p-1]):
		pnl[i_p] = pnl[i_p] - (buy_sell_fee * pnl[i_p])

# set up the figure
fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)

# initialise some labels for the plot
datenum_aug_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_aug_data]
datenum_price_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_price_data]

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
ax[2].grid(linewidth=0.4)
ax[0].plot(datenum_price_data, price_data, linewidth=0.5)
ax[1].plot(datenum_aug_data, sent_score, linewidth=0.5)
ax[2].plot(datenum_price_data, pnl, linewidth=0.5)

# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("Seniment score")
ax[2].set_ylabel("PnL")
ax[1].set_ylim([-5.5, 5.5])

ax[0].set_title("4_basic_strategy_example.py")

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d')
ax[0].xaxis.set_major_formatter(xfmt)

# show the plot
plt.show()
