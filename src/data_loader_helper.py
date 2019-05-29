import datetime
import pprint
import msgpack
import zlib

import io_helper as ioh
import datetime_helper as dh
import load_augmento_data_helper as ladh
import load_binance_data_helper as lbdh


def list_file_dates_for_path(path, filename_suffix, datetime_format_str):
	date_strs = ioh.list_files_in_path_os(path, filename_suffix=filename_suffix)
	date_strs = [el.split("/")[-1].replace(filename_suffix, "") for el in date_strs]
	dates = [dh.datetime_str_to_datetime(el, timestamp_format_str=datetime_format_str)
					for el in date_strs]
	return dates

def find_missing_date_batches(missing_days, required_days):
	missing_day_batches = []
	for i_amd in range(len(missing_days)):
		if i_amd > 0 and (missing_days[i_amd] - missing_days[i_amd-1]).days == 1:
			missing_day_batches[-1].append(missing_days[i_amd])
		else:
			missing_day_batches.append([missing_days[i_amd]])
	return missing_day_batches

def load_data(path_data="data/cache",
              augmento_coin=None,
              augmento_source=None,
              binance_symbol=None,
              dt_bin_size=None,
              datetime_start=None,
              datetime_end=None,
              augmento_api_key=None):
	
	# check the input arguments
	if None in [binance_symbol, augmento_coin, augmento_source, dt_bin_size, datetime_start, datetime_end]:
		raise Exception("missing required param(s) in load_data()")
	
	# specify the path for the binance data cache
	path_augmento_data = "{:s}/augmento/{:s}/{:s}/{:d}".format(*(path_data, augmento_source, augmento_coin, dt_bin_size))
	
	# specify the path for the augmento data cache
	path_binance_data = "{:s}/binance/{:s}/{:d}".format(*(path_data, binance_symbol, dt_bin_size))
	
	# make sure all the paths exist
	ioh.check_path(path_augmento_data, create_if_not_exist=True)
	ioh.check_path(path_binance_data, create_if_not_exist=True)
	
	# check which days of data exist for the augmento data and binance data
	augmento_dates = list_file_dates_for_path(path_augmento_data, ".msgpack.zlib", "%Y%m%d")
	binance_dates = list_file_dates_for_path(path_binance_data, ".msgpack.zlib", "%Y%m%d")
	
	# remove any dates from the last 3 days, so we reload of recent data
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
	
	quit()
	
	"""
	pprint.pprint(augmento_missing_batches)
	print("")
	pprint.pprint(binance_missing_batches)
	"""




