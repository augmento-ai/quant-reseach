import sys
import requests
import datetime
import time
import zlib
import msgpack

# import files from src
sys.path.insert(0, "src")
import io_helper as ioh
import datetime_helper as dh

# define where we're going to save the data
path_save_data = "data/example_data"
filename_save_data = "{:s}/bitmex_data.msgpack.zlib".format(path_save_data)

# define the start and end times
datetime_start = datetime.datetime(2018, 6, 1)
datetime_end = datetime.datetime(2019, 1, 1)

# initialise a store for the data we're downloading
market_data = []

# define a start pointer to track multiple requests
start_ptr = 0
count_ptr = 100

# get the data
while start_ptr >= 0:
	
	# define the url of the endpoint
	endpoint_url = "https://www.bitmex.com/api/v1/trade/bucketed"
	
	# define the parameters of the request
	params = {
		"symbol" : "XBt",
		"binSize" : "1h",
		"count" : count_ptr,
		"start" : start_ptr,
		"startTime" : datetime_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
		"endTime" : datetime_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
	}
	
	# make the request
	r = requests.request("GET", endpoint_url, params=params, timeout=10)
	
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
	
	# convert the iso timestamps to epoch times
	for td in temp_data:
		t_epoch = dh.timestamp_to_epoch(td["timestamp"], "%Y-%m-%dT%H:%M:%S.000Z")
		td.update({"t_epoch" : t_epoch})
	
	# extend the data store
	market_data.extend(temp_data)
	
	# print the progress
	str_print = "got data from {:s} to {:s}".format(*(market_data[0]["timestamp"],
	                                                market_data[-1]["timestamp"],))
	print(str_print)
	
	# sleep
	time.sleep(2.1)

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save the data
print("saving data to {:s}".format(filename_save_data))
with open(filename_save_data, "wb") as f:
	f.write(zlib.compress(msgpack.packb(market_data)))

print("done!")

