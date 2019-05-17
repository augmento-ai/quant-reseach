import numpy as np

def safe_divide(arr_a, arr_b):
	return np.divide(arr_a, arr_b, out=np.zeros_like(arr_a), where=(arr_b != 0))

def strip_data_by_time(t_data, data, t_min, t_max):
	data = np.array([s for s, t in zip(data, t_data) if t >= t_min and t <= t_max])
	t_data = np.array([t for t in t_data if t >= t_min and t <= t_max])
	return t_data, data

def causal_rolling_average(arr, window_size):
	filt = np.ones(window_size) / float(window_size)
	return np.convolve(arr, filt, mode="FULL")[0:arr.shape[0]]

def causal_rolling_sd(arr, window_size):
	new_arr = np.hstack((np.zeros(window_size-1), arr))
	out_arr = np.zeros(arr.shape[0])
	for i in range(out_arr.shape[0]):
		num = new_arr[i+window_size-1] - np.mean(new_arr[i : i + window_size-1])
		denom = np.std(new_arr[i : i + window_size-1])
		out_arr[i] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
	return out_arr

def compare_sents(sent_a, sent_b, window_size=1):
	
	# find the ratio
	s = np.divide(sent_a, sent_b, out=np.zeros_like(sent_a), where=(sent_b != 0))
	
	# average the data
	if window_size > 1:
		s = causal_rolling_average(s, window_size)
	
	return s

