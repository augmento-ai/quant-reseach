import example_helper as eh
import numpy as np
import math 



# TODO: improve commenting

class Data():

    def __init__(self):

        pass

    def load_raw(self, 
            augmento_topic = "data/example_data/augmento_topics.msgpack.zlib",
            augmento_data = "data/example_data/augmento_data.msgpack.zlib",
            bitmex_data = "data/example_data/bitmex_data.msgpack.zlib"):
        
        # load all raw data
        self.aug_topics, self.aug_topics_inv, self.t_aug_data,\
                self.aug_data, self.t_price_data, self.price_data =\
                eh.load_example_data(augmento_topic, augmento_data, bitmex_data)
        print("loaded")

        
    def get_data(self, n_timesteps, forward):

        # number of sentiments
        n_sentiments = self.aug_data.shape[1]
        
        # number of all data points
        n_data = self.aug_data.shape[0]
        
        # index of the last observation
        last_data = n_data - forward

        # number of all samples
        n_samples = last_data - n_timesteps  + 1

        # create empty arrays for sentiment and price
        arr_aug = np.zeros((n_samples, n_timesteps, n_sentiments),dtype=np.float64)
        
        #arr_price = np.zeros(n_samples)
        arr_price_full = np.zeros((n_samples, forward),dtype=np.float64)
        
        print("Loading...")
        for i in range(n_samples):
            arr_aug[i, :, :] = self.aug_data[i : i + n_timesteps,:]
            price_range = self.price_data[i + n_timesteps : i + n_timesteps + forward]
            #arr_price[i] = (price_range[-1]-price_range[0])/price_range[0]
            arr_price_full[i, :] = price_range 
            #print(arr_aug[i])
            #print(i)
            #print(n_samples)
        print("Ready.")
        
        self.arr_aug = arr_aug
        self.arr_price_full = arr_price_full

        #return arr_aug, arr_price_full


    def get_data_batch(self, batch_size):

        all_sentiment = self.arr_aug
        all_price = self.arr_price_full
        n_timesteps = all_price.shape[1]
        forward = all_price.shape[1]
        n_sentiments = all_sentiment.shape[2]
        n_pop = all_sentiment.shape[0]
        batch_sentiment = np.zeros((batch_size, n_timesteps, n_sentiments), dtype=np.float64)
        #batch_price = np.zeros(batch_size)
        batch_price = np.zeros((batch_size,forward), dtype=np.float64)
        batch_sequence = np.random.choice(n_pop, batch_size, replace=False)
        
        for i in range(len(batch_sequence)):
            batch_sentiment[i] = all_sentiment[batch_sequence[i]]
            batch_price[i] = all_price[batch_sequence[i]]

        return batch_sentiment, batch_price


