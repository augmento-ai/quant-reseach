import msgpack
import zlib
import numpy as np
import helper_functions as hf
import datetime_helper as dh

def strip_data_by_time(t_data, data, t_min, t_max):
	data = np.array([s for s, t in zip(data, t_data) if t >= t_min and t <= t_max])
	t_data = np.array([t for t in t_data if t >= t_min and t <= t_max])
	return t_data, data

def load_example_data(filename_augmento_topics,
                      filename_augmento_data,
                      filename_bitmex_data,
                      datetime_start=None,
                      datetime_end=None):

	# load the topics
	with open(filename_augmento_topics, "rb") as f:
		temp = msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8')
		augmento_topics = {int(k) : v for k, v in temp.items()}
		augmento_topics_inv = {v : int(k) for k, v in temp.items()}
	
	# load the augmento data
	with open(filename_augmento_data, "rb") as f:
		temp = msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8')
		t_aug_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
		aug_data = np.array([el["counts"] for el in temp], dtype=np.int32)
	
	# load the price data
	with open(filename_bitmex_data, "rb") as f:
		temp = msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8')
		t_price_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
		price_data = np.array([el["open"] for el in temp], dtype=np.float64)
	
	# set the start and end times if they are specified
	if datetime_start != None:
		t_start = dh.datetime_to_epoch(datetime_start)
	else:
		t_start = max(np.min(t_aug_data), np.min(t_price_data))
	
	if datetime_end != None:
		t_end = dh.datetime_to_epoch(datetime_end)
	else:
		t_end = min(np.max(t_aug_data), np.max(t_price_data))
	
	# strip the sentiments and prices outside the shared time range
	t_aug_data, aug_data = strip_data_by_time(t_aug_data, aug_data, t_start, t_end)
	t_price_data, price_data = strip_data_by_time(t_price_data, price_data, t_start, t_end)
	
	return augmento_topics, augmento_topics_inv, t_aug_data, aug_data, t_price_data, price_data
