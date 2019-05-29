import sys
import msgpack
import zlib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pprint

# import files from src
sys.path.insert(0, "src")
import example_helper as eh
import analysis_helper as ah
import data_loader_helper as dlh

# define the start and end times
datetime_start = datetime.datetime(2018, 10, 8)
datetime_end = datetime.datetime(2018, 12, 3)
#datetime_start = datetime.datetime(2019, 4, 8)
#datetime_end = datetime.datetime(2019, 4, 10)

all_data = dlh.load_data(path_data="data/cache",
              augmento_coin="bitcoin",
              augmento_source="twitter",
              binance_symbol="BTCUSDT",
              dt_bin_size=3600,
              datetime_start=datetime_start,
              datetime_end=datetime_end,
              augmento_api_key=None)
t_aug_data, t_bin_data, aug_data, bin_data, aug_keys, bin_keys = all_data



# set up the figure
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)

# initialise some labels for the plot
datenum_bin_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_bin_data]
datenum_aug_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_aug_data]

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
ax[0].plot(datenum_bin_data, bin_data[:, bin_keys["close"]], linewidth=0.5)
ax[1].plot(datenum_aug_data, aug_data, linewidth=0.5, alpha=0.5)

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax[0].xaxis.set_major_formatter(xfmt)

# show the plot
plt.show()



