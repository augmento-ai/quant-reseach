import sys
import msgpack
import zlib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md

# import files from src
sys.path.insert(0, "src")
import helper_functions as hf

# define the location of the input file
filename_augmento_topics = "data/example_data/augmento_topics.msgpack.zlib"
filename_augmento_data = "data/example_data/augmento_data.msgpack.zlib"
filename_bitmex_data = "data/example_data/bitmex_data.msgpack.zlib"

# load the topics
with open(filename_augmento_topics, "rb") as f:
	temp = hf.decode_bytes(msgpack.unpackb(zlib.decompress(f.read())))
	augmento_topics = {int(k) : v for k, v in temp.items()}
	augmento_topics_inv = {v : int(k) for k, v in temp.items()}

# load the augmento data
with open(filename_augmento_data, "rb") as f:
	temp = hf.decode_bytes(msgpack.unpackb(zlib.decompress(f.read())))
	t_augmento_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
	augmento_data = np.array([el["counts"] for el in temp], dtype=np.int32)

# load the price data
with open(filename_bitmex_data, "rb") as f:
	temp = hf.decode_bytes(msgpack.unpackb(zlib.decompress(f.read())))
	t_price_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
	price_data = np.array([el["close"] for el in temp], dtype=np.float64)

# get the signals we're interested in
augmento_signal_a = augmento_data[:, augmento_topics_inv["Positive"]]
augmento_signal_b = augmento_data[:, augmento_topics_inv["Negative"]]

# plot some data!
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)

# initialise some labels for the plot
datenum_augmento_data = [md.date2num(datetime.datetime.fromtimestamp(el))
							for el in t_augmento_data]
datenum_price_data = [md.date2num(datetime.datetime.fromtimestamp(el))
							for el in t_price_data]

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
ax[0].plot(datenum_price_data, price_data, linewidth=0.5)
ax[1].plot(datenum_augmento_data, augmento_signal_a, color="g", linewidth=0.5)
ax[1].plot(datenum_augmento_data, augmento_signal_b, color="r", linewidth=0.5)

# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("Seniments")

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax[0].xaxis.set_major_formatter(xfmt)

# show the plot
plt.show()


