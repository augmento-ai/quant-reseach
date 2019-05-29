import requests
import time
import pprint
import zlib
import msgpack
import math
import numpy as np

import datetime_helper as dh

# define the base url of the endpoint
base_url = "https://api.binance.com"

bin_size_keys = {
	3600 : "1h",
	86400 : "1d",
}

candle_dict = {
	0 : "t_open",
    1 : "open",
    2 : "high",
    3 : "low",
    4 : "close",
    5 : "volume",
    6 : "t_close",
    7 : "quote_asset_volume",
    8 : "trades",
    9 : "taker_buy_base_asset_volume",
    10 : "taker_buy_quote_asset_volume",
    11 : "ignore",
}

output_keys = ["open", "close", "high", "low", "volume", "trades"]

def load_keys():
	return {v : i for i, v in enumerate(output_keys)}

def load_and_cache_data(path_output, symbol, dt_bin_size, datetime_start, datetime_end):
	
	# make sure the start date and end date are rounded to the nearest day
	datetime_start = dh.round_datetime_to_day_start(datetime_start)
	datetime_end = dh.round_datetime_to_day_start(datetime_end)
	
	# check that the dt_bin_size is valid
	if dt_bin_size not in bin_size_keys.keys():
		raise Exception("invalid binance bin_size: {:s} not in: {:s}".format(*(dt_bin_size, bin_size_keys.keys())))
	
	# get the start and end times in seconds
	t_epoch_start = int(dh.datetime_to_epoch(datetime_start))
	t_epoch_end = int(dh.datetime_to_epoch(datetime_end)) - dt_bin_size
	
	# set the max number of results per request
	n_limit = 1000
	
	# create a list of start and end times
	n_requests = math.ceil(float(t_epoch_end - t_epoch_start) / float(dt_bin_size) / float(n_limit))
	dt_request_starts = [(i_r * n_limit * dt_bin_size) for i_r in range(n_requests)]
	dt_request_ends = [((i_r + 1) * n_limit * dt_bin_size) - dt_bin_size for i_r in range(n_requests)]
	t_request_starts = [min(dt_r + t_epoch_start, t_epoch_end) for dt_r in dt_request_starts]
	t_request_ends = [min(dt_r + t_epoch_start, t_epoch_end) for dt_r in dt_request_ends]
	
	# initialise a store for the candles
	candles = []
	
	# go through the list of start and end times and make the requests
	for t_s, t_e in zip(t_request_starts, t_request_ends):
	
		# set up the parameters for the request
		params = {
			"symbol" : symbol,
			"interval" : bin_size_keys[dt_bin_size],
			"startTime" : t_s * 1000,
			"endTime" : t_e * 1000,
			"limit" : n_limit,
		}
		
		# make a request
		r = requests.request("GET", "{:s}/api/v1/klines".format(base_url), params=params, timeout=10)
		temp_candles = r.json()
		
		# print the progress
		first_datetime = dh.epoch_to_datetime(temp_candles[0][0] / 1000)
		last_datetime = dh.epoch_to_datetime(temp_candles[-1][0] / 1000)
		str_print = "got binance data from {:s} to {:s}".format(*(str(first_datetime),
		                                                        str(last_datetime)))
		print(str_print)
		
		# if the request was ok, add the data
		# else return an error
		if r.status_code == 200:
			candles.extend(temp_candles)
		else:
			raise Exception("api call failed with status_code {:d}".format(r.status_code))
		
		time.sleep(2.0)
	
	# make sure we got data between the two times
	if len(candles) == 0 or candles[0][0]/1000 != t_request_starts[0] or candles[-1][0]/1000 != t_request_ends[-1]:
		raise Exception("failed to get binance data for range")
	else:
		print("all good!")
	
	# fill in any missing candles
	full_candles = []
	for i_c, c in enumerate(candles):
		
		# append this candle to the full candles list
		full_candles.append(candles[i_c].copy())
		
		# if this isn't the last candle, append any missing candles following this one
		if i_c < len(candles) - 1:
			
			# while the next candle is missing, replace it with a copy of this candle
			while candles[i_c + 1][0] > full_candles[-1][0] + (dt_bin_size * 1000):
				blank_candle = full_candles[-1].copy()
				
				# TODO: make sure we change the other params here
				blank_candle[0] += dt_bin_size * 1000
				full_candles.append(blank_candle)
	
	# get datetimes for all datapoints
	datetimes = [dh.epoch_to_datetime(el[0] / 1000) for el in full_candles]
	
	# get the starts of all the days
	days = sorted(list(set([dh.round_datetime_to_day_start(el) for el in datetimes])))
	
	# for each of the start dates, cache the data for that day
	for day in days:
		
		# generate the output filename
		output_filename_short = dh.datetime_to_str(day, timestamp_format_str="%Y%m%d")
		output_filename = "{:s}/{:s}.msgpack.zlib".format(*(path_output, output_filename_short))
		
		# generate/filter the output data
		temp_start = dh.datetime_to_epoch(day)
		temp_end = dh.datetime_to_epoch(dh.add_days_to_datetime(day, 1))
		output_data = [el for el in full_candles if el[0] / 1000 >= temp_start and el[0] / 1000 < temp_end]
		
		# save the data
		with open(output_filename, "wb") as f:
			f.write(zlib.compress(msgpack.packb(output_data)))

def load_cached_data(path_input, datetime_start, datetime_end):
	
	# initialise the output data
	output_data = []
	
	# get a list of the files we need to open
	required_dates = dh.get_datetimes_between_datetimes(datetime_start, datetime_end)
	
	# go through all the dates and load the corrisponding files
	for rd in required_dates:
		
		# load the file
		input_filename_short = dh.datetime_to_str(rd, timestamp_format_str="%Y%m%d")
		input_filename = "{:s}/{:s}.msgpack.zlib".format(*(path_input, input_filename_short))
		with open(input_filename, "rb") as f:
			output_data.extend(msgpack.unpackb(zlib.decompress(f.read()), encoding='utf-8'))
	
	# format the data
	# [open, close, high, low, volume, trades]
	t_data = np.array([float(el[0]) / 1000.0 for el in output_data], dtype=np.float64)
	candle_dict_inv = {v : k for k, v in candle_dict.items()}
	key_indexes = [candle_dict_inv[k_i] for k_i in output_keys]
	feat_data = np.array([[float(el[i]) for i in key_indexes] for el in output_data], dtype=np.float64)
	
	return t_data, feat_data
