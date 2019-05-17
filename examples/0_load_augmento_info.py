import sys
import requests
import datetime
import time
import zlib
import msgpack

# import files from src
sys.path.insert(0, "src")
import helper_functions as hf
import io_helper as ioh

# define the urls of the endpoints for all the info
topics_endpoint_url = "http://api-dev.augmento.ai/v0.1/topics"
sources_endpoint_url = "http://api-dev.augmento.ai/v0.1/sources"
coins_endpoint_url = "http://api-dev.augmento.ai/v0.1/coins"
bin_sizes_endpoint_url = "http://api-dev.augmento.ai/v0.1/bin_sizes"

# define where we're going to save the data
path_save_data = "data/example_data"
filename_save_topics = "{:s}/augmento_topics.msgpack.zlib".format(path_save_data)
filename_save_sources = "{:s}/augmento_sources.msgpack.zlib".format(path_save_data)
filename_save_coins = "{:s}/augmento_coins.msgpack.zlib".format(path_save_data)
filename_save_bin_sizes = "{:s}/augmento_bin_sizes.msgpack.zlib".format(path_save_data)

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save a list of the augmento topics
r = requests.request("GET", topics_endpoint_url, timeout=10)
print("saving topics to {:s}".format(filename_save_topics))
with open(filename_save_topics, "wb") as f:
	f.write(zlib.compress(msgpack.packb(r.json())))

# save a list of the augmento topics
r = requests.request("GET", sources_endpoint_url, timeout=10)
print("saving sources to {:s}".format(filename_save_sources))
with open(filename_save_sources, "wb") as f:
	f.write(zlib.compress(msgpack.packb(r.json())))

# save a list of the augmento topics
r = requests.request("GET", coins_endpoint_url, timeout=10)
print("saving coins to {:s}".format(filename_save_coins))
with open(filename_save_coins, "wb") as f:
	f.write(zlib.compress(msgpack.packb(r.json())))

# save a list of the augmento topics
r = requests.request("GET", bin_sizes_endpoint_url, timeout=10)
print("saving bin_sizes to {:s}".format(filename_save_bin_sizes))
with open(filename_save_bin_sizes, "wb") as f:
	f.write(zlib.compress(msgpack.packb(r.json())))


print("done!")

