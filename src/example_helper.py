import msgpack
import zlib
import numpy as np
import helper_functions as hf

def load_example_data(filename_augmento_topics,
                      filename_augmento_data,
                      filename_bitmex_data):

	# load the topics
	with open(filename_augmento_topics, "rb") as f:
		temp = msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8')
		augmento_topics = {int(k) : v for k, v in temp.items()}
		augmento_topics_inv = {v : int(k) for k, v in temp.items()}
	
	# load the augmento data
	with open(filename_augmento_data, "rb") as f:
		temp = msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8')
		t_augmento_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
		augmento_data = np.array([el["counts"] for el in temp], dtype=np.int32)
	
	# load the price data
	with open(filename_bitmex_data, "rb") as f:
		temp = msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8')
		t_price_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
		price_data = np.array([el["close"] for el in temp], dtype=np.float64)
	
	return augmento_topics, augmento_topics_inv, t_augmento_data, augmento_data, t_price_data, price_data