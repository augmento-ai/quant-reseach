import datetime
import pprint
import msgpack
import zlib
import numpy as np

import io_helper as ioh
import datetime_helper as dh
import load_augmento_data_helper as ladh
#import load_binance_data_helper as lbdh
import load_kraken_data_helper as lbdh


def find_missing_date_batches(missing_days, required_days):
	missing_day_batches = []
	for i_amd in range(len(missing_days)):
		if i_amd > 0 and (missing_days[i_amd] - missing_days[i_amd-1]).days == 1:
			missing_day_batches[-1].append(missing_days[i_amd])
		else:
			missing_day_batches.append([missing_days[i_amd]])
	return missing_day_batches

def strip_data_by_time(t_data, data, t_min, t_max):
	data = np.array([s for s, t in zip(data, t_data) if t >= t_min and t <= t_max])
	t_data = np.array([t for t in t_data if t >= t_min and t <= t_max])
	return t_data, data

def load_data(path_data="data/cache",
              augmento_coin=None,
              augmento_source=None,
              binance_symbol=None,
              dt_bin_size=None,
              datetime_start=None,
              datetime_end=None,
              augmento_api_key=None):
	
	datetime_end = min(datetime.datetime.now(), datetime_end)
	
	# check the input arguments
	if None in [binance_symbol, augmento_coin, augmento_source, dt_bin_size, datetime_start, datetime_end]:
		raise Exception("missing required param(s) in load_data()")
	
	# specify the path for the binance data cache
	path_augmento_data = "{:s}/augmento/{:s}/{:s}/{:d}".format(*(path_data, augmento_source, augmento_coin, dt_bin_size))
	path_augmento_topics = "{:s}/augmento/".format(path_data)
	
	# specify the path for the augmento data cache
	#path_binance_data = "{:s}/binance/{:s}/{:d}".format(*(path_data, binance_symbol, dt_bin_size))
	path_binance_data = "{:s}/kraken/{:s}/{:d}".format(*(path_data, binance_symbol, dt_bin_size))
	
	# make sure all the paths exist
	ioh.check_path(path_augmento_data, create_if_not_exist=True)
	ioh.check_path(path_binance_data, create_if_not_exist=True)
	
	# check which days of data exist for the augmento data and binance data
	augmento_dates = dh.list_file_dates_for_path(path_augmento_data, ".msgpack.zlib", "%Y%m%d")
	binance_dates = dh.list_file_dates_for_path(path_binance_data, ".msgpack.zlib", "%Y%m%d")
	
	# remove any dates from the last 3 days, so we reload recent data
	datetime_now = datetime.datetime.now()
	augmento_dates = [el for el in augmento_dates if el < dh.add_days_to_datetime(datetime_now, -3)]
	binance_dates = [el for el in binance_dates if el < dh.add_days_to_datetime(datetime_now, -3)]
	
	# get a list of the days we need
	required_dates = dh.get_datetimes_between_datetimes(datetime_start, datetime_end)
	
	# get a list of the days we're missing for augmento and binance data
	augmento_missing_dates = sorted(list(set(required_dates) - set(augmento_dates)))
	binance_missing_dates = sorted(list(set(required_dates) - set(binance_dates)))
	
	# group the missing days by batch
	augmento_missing_batches = find_missing_date_batches(augmento_missing_dates, required_dates)
	binance_missing_batches = find_missing_date_batches(binance_missing_dates, required_dates)
	
	# load the augmento keys
	aug_keys = ladh.load_keys(path_augmento_topics)
	
	# load the binance keys
	bin_keys = lbdh.load_keys()
	
	# for each of the missing batches of augmento data, get the data and cache it
	for abds in augmento_missing_batches:
		
		# get the data for the batch and cache it
		ladh.load_and_cache_data(path_augmento_data,
		                                          augmento_source,
		                                          augmento_coin,
		                                          dt_bin_size,
		                                          abds[0],
		                                          dh.add_days_to_datetime(abds[-1], 1))
	
	# for each of the missing batches of binance data, get the data and cache it
	for bbds in binance_missing_batches:
		
		# get the data for the batch and cache it
		lbdh.load_and_cache_data(path_binance_data,
		                                          binance_symbol,
		                                          dt_bin_size,
		                                          bbds[0],
		                                          dh.add_days_to_datetime(bbds[-1], 1))
	
	# load the data
	t_aug_data, aug_data = ladh.load_cached_data(path_augmento_data, datetime_start, datetime_end)
	t_bin_data, bin_data = lbdh.load_cached_data(path_binance_data, datetime_start, datetime_end)
	
	# strip the data
	t_min = max([t_aug_data[0], t_bin_data[0], dh.datetime_to_epoch(datetime_start)])
	t_max = min([t_aug_data[-1], t_bin_data[-1], dh.datetime_to_epoch(datetime_end)])
	t_aug_data, aug_data = strip_data_by_time(t_aug_data, aug_data, t_min, t_max)
	t_bin_data, bin_data = strip_data_by_time(t_bin_data, bin_data, t_min, t_max)
	
	return t_aug_data, t_bin_data, aug_data, bin_data, aug_keys, bin_keys




