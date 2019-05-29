import requests
import time
import pprint
import zlib
import msgpack

import datetime_helper as dh

# define the base url of the endpoint
base_url = "http://api-dev.augmento.ai/v0.1"

def load_and_cache_data(path_output, source, coin, bin_size, datetime_start, datetime_end):
	
	# make sure the start date and end date are rounded to the nearest day
	datetime_start = dh.round_datetime_to_day_start(datetime_start)
	datetime_end = dh.round_datetime_to_day_start(datetime_end)
	
	# make sure the source exists
	available_sources = requests.request("GET", "{:s}/sources".format(base_url), timeout=10).json()
	if source not in available_sources:
		raise Exception("invalid augmento source: {:s} not in: {:s}".format(*(source, available_sources)))
	
	# make sure the coin exists
	available_coins = requests.request("GET", "{:s}/coins".format(base_url), timeout=10).json()
	if coin not in available_coins:
		raise Exception("invalid augmento coin: {:s} not in: {:s}".format(*(coin, available_coins)))
	
	# make sure the bin_size exists
	available_bin_sizes = requests.request("GET", "{:s}/bin_sizes".format(base_url), timeout=10).json()
	available_bin_sizes = {v : k for k, v in available_bin_sizes.items()}
	if bin_size not in available_bin_sizes:
		raise Exception("invalid augmento bin_size: {:s} not in: {:s}".format(*(bin_size, available_bin_sizes)))
	
	# initialise a store for the data we're downloading
	sentiment_data = []

	# define a start pointer to track multiple requests
	start_ptr = 0
	count_ptr = 1000

	# get the data
	while start_ptr >= 0:
		
		# define the parameters of the request
		params = {
			"source" : source,
			"coin" : coin,
			"bin_size" : available_bin_sizes[bin_size],
			"count_ptr" : count_ptr,
			"start_ptr" : start_ptr,
			"start_datetime" : datetime_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
			"end_datetime" : datetime_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
		}
		
		# make the request
		r = requests.request("GET", "{:s}/events/aggregated".format(base_url), params=params, timeout=10)
		
		# if the request was ok, add the data and increment the start_ptr
		# else return an error
		if r.status_code == 200:
			temp_data = r.json()
			start_ptr += count_ptr
		else:
			raise Exception("api call failed with status_code {:d}".format(r.status_code))
		
		# if we didn't get any data, assume we've got all the data
		if len(temp_data) == 0:
			start_ptr = -1
		
		# extend the data store
		sentiment_data.extend(temp_data)
		
		# print the progress
		str_print = "got augmento data from {:s} to {:s}".format(*(sentiment_data[0]["datetime"],
		                                                         sentiment_data[-1]["datetime"],))
		print(str_print)
		
		# sleep
		time.sleep(2.0)
	
	# get datetimes for all datapoints
	datetimes = [dh.epoch_to_datetime(el["t_epoch"]) for el in sentiment_data]
	
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
		output_data = [el for el in sentiment_data if el["t_epoch"] >= temp_start and el["t_epoch"] < temp_end]
		
		# save the data
		with open(output_filename, "wb") as f:
			f.write(zlib.compress(msgpack.packb(output_data)))
	
