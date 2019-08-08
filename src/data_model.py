import example_helper as eh
import numpy as np

class Data():

    def __init__(self):
        self.filename_augmento_topics = "data/example_data/augmento_topics.msgpack.zlib"
        self.filename_augmento_data = "data/example_data/augmento_data.msgpack.zlib"
        self.filename_bitmex_data = "data/example_data/bitmex_data.msgpack.zlib"
        self.all_data = eh.load_example_data(self.filename_augmento_topics,
                self.filename_augmento_data,
                self.filename_bitmex_data)
    
        
    def load_raw(self, 
            augmento_topic = "data/example_data/augmento_topics.msgpack.zlib",
            augmento_data = "data/example_data/augmento_data.msgpack.zlib",
            bitmex_data = "data/example_data/bitmex_data.msgpack.zlib"):
        
        # load all raw data
        self.aug_topics, self.aug_topics_inv, self.t_aug_data,\
                self.aug_data, self.t_price_data, self.price_data =\
                eh.load_example_data(augmento_topic, augmento_data, bitmex_data)
        print("loaded")
    def data(self, n_samples, n_timesteps, forward):

        # load data
        #aug_data = self.raw()[3]
        #price_data = self.raw()[5]

        # number of sentiments
        n_sentiments = self.aug_data.shape[1]
        
        # set max and min values for window size
        window_max = self.aug_data.shape[0]-forward
        window_min = n_samples
        
        # create empty arrays for sentiment and price data
        arr_aug = np.zeros((n_samples, n_timesteps, n_sentiments), dtype=np.float64)
        arr_price = np.zeros(n_samples)
        
        for i in range(n_samples):
            # set range for sensitment data
            end = round(np.random.uniform(n_timesteps-1, window_max))
            start = end - n_timesteps
            sample = self.aug_data[start:end,]
            # set location of price data
            price_start = end + forward
            # fill data
            arr_aug[i] = sample
            arr_price[i] = self.price_data[price_start]


        return arr_aug, arr_price

    def get_data(self, n_timesteps, forward):

        print("tbc")


    def get_data_batch(self, n_samples, n_timesteps, forward, batch_size):
        
        all_sentiment, all_price = self.data(n_samples, n_timesteps, forward)
        
        n_sentiments = all_sentiment.shape[2]
        n_pop = all_sentiment.shape[0]
        print(n_sentiments)
        batch_sentiment = np.zeros((batch_size, n_timesteps, n_sentiments), dtype=np.float64)
        batch_price = np.zeros(batch_size)
        
        batch_sequence = np.random.choice(n_pop, batch_size, replace=False)

        for i in range(len(batch_sequence)):
            batch_sentiment[i] = all_sentiment[batch_sequence[i]]
            batch_sequence[i] = all_price[batch_sequence[i]]

        return batch_sentiment, batch_sequence




