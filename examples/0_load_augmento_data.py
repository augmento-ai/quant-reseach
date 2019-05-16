import sys
import requests
import datetime
import time
import zlib
import msgpack

# import files from src
sys.path.insert(0, "src")
import io_helper as ioh

# define the url of the endpoint
endpoint_url = "http://54.76.10.107/v0.1"

# define where we're going to save the data
path_save_data = "data/example_data"
filename_save_data = "{:s}/augmento_data.msgpack.zlib".format(path_save_data)
filename_save_topics = "{:s}/augmento_topics.msgpack.zlib".format(path_save_data)

# define the start and end times
datetime_start = datetime.datetime(2018, 6, 1)
datetime_end = datetime.datetime(2019, 1, 1)

# initialise a store for the data we're downloading
sentiment_data = []

# define a start pointer to track multiple requests
start_ptr = 0
count_ptr = 100

# get the data
while start_ptr >= 0:
	
	# define the parameters of the request
	params = {
		"source" : "twitter",
		"coin" : "bitcoin",
		"bin_size" : "1H",
		"count_ptr" : count_ptr,
		"start_ptr" : start_ptr,
		"start_datetime" : datetime_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
		"end_datetime" : datetime_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
	}
	
	# make the request
	r = requests.request("GET", "{:s}/events/aggregated".format(endpoint_url),
	                     params=params, timeout=10)
	
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
	str_print = "got data from {:s} to {:s}".format(*(sentiment_data[0]["datetime"],
	                                                sentiment_data[-1]["datetime"],))
	print(str_print)
	
	# sleep
	time.sleep(1.0)

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save the data
print("saving data to {:s}".format(filename_save_data))
with open(filename_save_data, "wb") as f:
	f.write(zlib.compress(msgpack.packb(sentiment_data)))

# also request and save a list of the augmento topics
r = requests.request("GET", "{:s}/topics".format(endpoint_url), timeout=10)
print("saving data to {:s}".format(filename_save_data))
with open(filename_save_topics, "wb") as f:
	f.write(zlib.compress(msgpack.packb(r.json())))


print("done!")

