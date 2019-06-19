import numpy as np
import numba as nb


@nb.jit("(f8[:])(f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def nb_safe_divide(a, b):
	# divide each element in a by each element in b
	# if element b == 0.0, return element = 0.0
	c = np.zeros(a.shape[0], dtype=np.float64)
	for i in range(a.shape[0]):
		if b[i] != 0.0:
			c[i] = a[i] / b[i]
	return c

@nb.jit("(f8[:])(f8[:], i8)", nopython=True, nogil=True, cache=True)
def nb_causal_rolling_average(arr, window_size):
	
	# create an output array
	out_arr = np.zeros(arr.shape[0])
	
	# create an array from the input array, with added space for the rolling window
	new_arr = np.hstack((np.ones(window_size-1) * arr[0], arr))
	
	# for each output element, find the mean of the last few input elements
	for i in range(out_arr.shape[0]):
		out_arr[i] = np.mean(new_arr[i : i + window_size])
	
	return out_arr

@nb.jit("(f8[:])(f8[:], i8)", nopython=True, nogil=True, cache=True)
def nb_causal_rolling_sd(arr, window_size):
	
	# create an output array
	out_arr = np.zeros(arr.shape[0])
	
	# create an array from the input array, with added space for the rolling window
	new_arr = np.hstack((np.ones(window_size-1) * arr[0], arr))
	
	# for each output element, find the mean and std of the last few
	# input elements, and standardise the input element by the mean and std of the window
	for i in range(out_arr.shape[0]):
		num = new_arr[i+window_size-1] - np.mean(new_arr[i : i + window_size-1])
		denom = np.std(new_arr[i : i + window_size-1])
		if denom != 0.0:
			out_arr[i] = num / denom
	
	return out_arr
	
@nb.jit("(f8[:])(f8[:], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def nb_calc_sentiment_score_a(sent_a, sent_b, ra_win_size, std_win_size):
	# example method for creating a stationary sentiment score based on Augmento data
	
	# compare the raw sentiment values
	sent_ratio = nb_safe_divide(sent_a, sent_b)
	
	# smooth the sentiment ratio
	sent_ratio_smooth = nb_causal_rolling_average(sent_ratio, ra_win_size)
	
	# create a stationary(ish) representation of the smoothed sentiment ratio
	sent_score = nb_causal_rolling_sd(sent_ratio_smooth, std_win_size)
	
	return sent_score

@nb.jit("(f8[:])(f8[:], f8[:], f8, f8)", nopython=True, nogil=True, cache=True)
def nb_backtest_a(price, sent_score, start_pnl, buy_sell_fee):
	# example backtest with approximate model for long/short contracts
	
	# create an array to hold our pnl, and set the first value
	pnl = np.zeros(price.shape, dtype=np.float64)
	pnl[0] = start_pnl
	
	# for each step, run the market model
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
