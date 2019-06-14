import numpy as np
import numba as nb

def safe_divide(arr_a, arr_b):
	return np.divide(arr_a, arr_b, out=np.zeros_like(arr_a), where=(arr_b != 0))

def causal_rolling_average(arr, window_size):
	# find the mean of the past X samples (only look into the past)
	filt = np.ones(window_size) / float(window_size)
	return np.convolve(arr, filt, mode="FULL")[0:arr.shape[0]]

def causal_rolling_sd(arr, window_size):
	# standardise each sample w.r.t. the past X samples (only look into the past)
	
	# create an array to hold our rolling window, and an output array
	new_arr = np.hstack((np.zeros(window_size-1), arr))
	out_arr = np.zeros(arr.shape[0])
	
	# for each output...
	for i in range(out_arr.shape[0]):
		
		# find the numerator and denom. for equ. x_sd = (x_i - mean_w) / sigma_w
		num = new_arr[i+window_size-1] - np.mean(new_arr[i : i + window_size-1])
		denom = np.std(new_arr[i : i + window_size-1])
		
		# find the sample standardised w.r.t. the past window
		out_arr[i] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
	return out_arr

def compare_sents(sent_a, sent_b, window_size=1):
	
	# find the ratio
	s = np.divide(sent_a, sent_b, out=np.zeros_like(sent_a), where=(sent_b != 0))
	
	# average the data
	if window_size > 1:
		s = causal_rolling_average(s, window_size)
	
	return s

@nb.jit("(f8[:])(f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def nb_safe_divide(a, b):
	c = np.zeros(a.shape[0], dtype=np.float64)
	for i in range(a.shape[0]):
		if b[i] != 0.0:
			c[i] = a[i] / b[i]
	return c

@nb.jit("(f8[:])(f8[:], i8)", nopython=True, nogil=True, cache=True)
def nb_causal_rolling_average(arr, window_size):
	# create an array to hold our rolling window, and an output array
	new_arr = np.hstack((np.zeros(window_size-1), arr))
	out_arr = np.zeros(arr.shape[0])
	for i in range(arr.shape[0]):
		out_arr[i] = np.mean(new_arr[i : i + window_size])
	return out_arr

@nb.jit("(f8[:])(f8[:], i8)", nopython=True, nogil=True, cache=True)
def nb_causal_rolling_sd(arr, window_size):
	# create an array to hold our rolling window, and an output array
	new_arr = np.hstack((np.zeros(window_size-1), arr))
	out_num_arr = np.zeros(arr.shape[0])
	out_den_arr = np.zeros(arr.shape[0])
	out_arr = np.zeros(arr.shape[0])
	for i in range(arr.shape[0]):
		out_num_arr[i] = new_arr[i+window_size-1] - np.mean(new_arr[i : i + window_size-1])
		out_den_arr[i] = np.std(new_arr[i : i + window_size-1])
	out_arr = nb_safe_divide(out_num_arr, out_den_arr)
	return out_arr
	
@nb.jit("(f8[:])(f8[:], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def nb_calc_sentiment_score_a(sent_a, sent_b, ra_win_size, std_win_size):
	sent_ratio = nb_safe_divide(sent_a, sent_b)
	sent_ratio_smooth = nb_causal_rolling_average(sent_ratio, ra_win_size)
	sent_score = nb_causal_rolling_sd(sent_ratio_smooth, std_win_size)
	return sent_score

@nb.jit("(f8[:])(f8[:], f8[:], f8)", nopython=True, nogil=True, cache=True)
def nb_backtest_a(price, sent_score, buy_sell_fee):
	pnl = np.zeros(price.shape, dtype=np.float64)
	pnl[0] = 1.0
	for i_p in range(1, price.shape[0]):
		
		# if sentiment score is positive, simulate long position
		# else if sentiment score is negative, simulate short position
		# else if the sentiment score is 0.0, hold
		# (note that this is a very approximate market simulation!)
		if sent_score[i_p-1] > 0.0:
			pnl[i_p] = (price[i_p] / price[i_p-1]) * pnl[i_p-1]
		elif sent_score[i_p-1] <= 0.0:
			pnl[i_p] = (price[i_p-1] / price[i_p]) * pnl[i_p-1]
		elif sent_score[i_p-1] == 0.0:
			pnl[i_p] = pnl[i_p-1]
		
		# simulate a trade fee if we cross from long to short, or visa versa
		if np.sign(sent_score[i_p]) != np.sign(sent_score[i_p-1]):
			pnl[i_p] = pnl[i_p] - (buy_sell_fee * pnl[i_p])
	
	return pnl
