import datetime
import io_helper as ioh

# set the utc value of the epoch
epoch = datetime.datetime.utcfromtimestamp(0)

def date_str_to_seconds(date_str, format):
	return (datetime.datetime.strptime(date_str, format) - datetime.datetime(1970,1,1)).total_seconds()

def datetime_str_to_datetime(date_str, timestamp_format_str="%Y-%m-%d %H:%M:%S"):
	return datetime.datetime.strptime(date_str, timestamp_format_str)

def timestamp_to_epoch(timestamp_str, timestamp_format_str):
	return (datetime.datetime.strptime(timestamp_str, timestamp_format_str) - epoch).total_seconds()

def epoch_to_datetime(t_epoch):
	# must be UTC to remove timezone
	return datetime.datetime.utcfromtimestamp(t_epoch)

def epoch_to_datetime_str(t_epoch, timestamp_format_str="%Y-%m-%d %H:%M:%S"):
	# must be UTC to remove timezone
	return datetime.datetime.utcfromtimestamp(t_epoch).strftime(timestamp_format_str)

def datetime_to_str(datetime_a, timestamp_format_str="%Y-%m-%d %H:%M:%S"):
	# must be UTC to remove timezone
	return datetime_a.strftime(timestamp_format_str)

def round_datetime_to_day_start(datetime_a, forward_days=0):
	datetime_a = datetime_a.replace(hour=0, minute=0, second=0, microsecond=0)
	return add_days_to_datetime(datetime_a, forward_days)

def add_days_to_datetime(datetime_a, forward_days):
	return datetime_a + datetime.timedelta(days=forward_days)

def datetime_to_epoch(datetime_a):
	return (datetime_a - epoch).total_seconds()

def timestamp_to_datetime(timestamp_str, timestamp_format_str):
	return datetime.datetime.strptime(timestamp_str, timestamp_format_str)

def list_file_dates_for_path(path, filename_suffix, datetime_format_str):
	date_strs = ioh.list_files_in_path_os(path, filename_suffix=filename_suffix)
	date_strs = [el.split("/")[-1].replace(filename_suffix, "") for el in date_strs]
	dates = [datetime_str_to_datetime(el, timestamp_format_str=datetime_format_str)
					for el in date_strs]
	return dates

def get_datetimes_between_datetimes(datetime_start, datetime_end):
	#return [datetime_start + datetime.timedelta(days=x)
	#		for x in range(0, (datetime_end-datetime_start).days + 1)]
	round_datetime_start = round_datetime_to_day_start(datetime_start)
	round_datetime_end = round_datetime_to_day_start(datetime_end)
	return [round_datetime_start + datetime.timedelta(days=x)
			for x in range(0, (round_datetime_end-round_datetime_start).days + 1)]
