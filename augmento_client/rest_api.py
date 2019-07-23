#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-07-22 15:03:30
# @Last modified by: ArthurBernard
# @Last modified time: 2019-07-23 14:25:41

""" Client connector to Augmento REST API. """

# Built-in packages
import logging
import json
import time
import datetime

# External packages
import requests
import pandas as pd

# Local packages

__all__ = ['RequestAugmento']


class RequestAugmento:
    """ Class to request Augmento data from REST public API.

    Methods
    -------
    send_request(method, **params)
        Return answere of request in list or dict.
    get_data(source, coin, bin_size, start, end, start_ptr=0, count_ptr=1000)
        Return aggregated event data in list of list.
    get_dataframe(source, coin, bin_size, start, end)
        Return aggregated event in a dataframe.
    get_database(source, coin, bin_size, start, end)
        Merge several requests of aggregated event in a dataframe.

    """

    def __init__(self, url='http://api-dev.augmento.ai/v0.1/',
                 logging_level='WARNING'):
        """ Initialize object. """
        self.url = url
        self.logger = logging.getLogger('get_augmento_data.' + __name__)
        self.logger.setLevel(logging_level)
        self.logger.debug('Starting augmento client')

    def send_request(self, method, **params):
        """ Send a request to Augmento REST public API.

        Parameters
        ----------
        method : str
            Name of the relevent request.
        **params : dict
            Relevent parameters, cf augemento documentation [1]_.

        Returns
        -------
        dict
            Relevant data.

        References
        ----------
        .. [1] http://api-dev.augmento.ai/v0.1/documentation#introduction

        """
        self.logger.debug(f'{method} request with {params} parameters.')

        # Try and catch some exceptions
        try:
            ans = requests.get(self.url + method, params)

            return json.loads(ans.text)

        except json.decoder.JSONDecodeError:
            self.logger.error('JSON error.')
            time.sleep(1)

            return self.send_request(method, **params)

        except requests.exceptions.ConnectionError:
            self.logger.error('HTTP error.')
            time.sleep(1)

            return self.send_request(method, **params)

        except Exception as e:
            self.logger.error('Unknown error {}.'.format(type(e)),
                              exc_info=True)
            time.sleep(1)

            return self.send_request(method, **params)

    def get_data(self, source, coin, bin_size, start, end, start_ptr=0,
                 count_ptr=1000):
        """ Request data to Augmento REST public API.

        Parameters
        ----------
        source : str, {'bitcointalk', 'reddit', 'twitter'}
            Source of data.
        coin : str
            Name of a crypto-currency, cf augemento documentation [1]_.
        bin_size : str, {'1H', '24H'}
            Time between two observations.
        start, end : str, int or datetime
            Starting date and ending date. If string must be ISO 8601 format
            such that ('%Y-%m-%dT%H:%M:%SZ'), or if integer must be UTC
            timestamp, else can be a datetime object.
        start_ptr : int, optional
            Default is 0.
        count_ptr : int, optional
            Number of observation.

        Returns
        -------
        list of list
            Relevant data from `date_0` to `date_T` as
            `[[x_1, ..., date_0, ts_0], ..., [x_1, ..., date_T, ts_T]]`.

        References
        ----------
        .. [1] http://api-dev.augmento.ai/v0.1/documentation#introduction

        """
        start = intel_date(start)
        end = intel_date(end)

        # Request data
        data = self.send_request(
            'events/aggregated', source=source, coin=coin, bin_size=bin_size,
            start_datetime=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_datetime=end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            start_ptr=start_ptr, count_ptr=count_ptr
        )

        return [[*x['counts'], x['datetime'], x['t_epoch']] for x in data]

    def get_dataframe(self, source, coin, bin_size, start, end):
        """ Request data to Augmento REST public API.

        Parameters
        ----------
        source : str, {'bitcointalk', 'reddit', 'twitter'}
            Source of data.
        coin : str
            Name of a crypto-currency, cf augemento documentation [1]_.
        bin_size : str, {'1H', '24H'}
            Time between two observations.
        start, end : str, int or datetime
            Starting date and ending date. If string must be ISO 8601 format
            such that ('%Y-%m-%dT%H:%M:%SZ'), or if integer must be UTC
            timestamp, else can be a datetime object.
            Warning : `end` and `start` must have less than 1000 observations
            between.


        Returns
        -------
        pd.DataFrame
            Relevant dataframe.

        References
        ----------
        .. [1] http://api-dev.augmento.ai/v0.1/documentation#introduction

        """
        # Request data
        data = self.get_data(
            source=source, coin=coin, bin_size=bin_size, start=start, end=end
        )

        return self._set_dataframe(data)

    def get_database(self, source, coin, bin_size, start, end):
        """ Merge several data request to Augmento REST public API.

        Parameters
        ----------
        source : str, {'bitcointalk', 'reddit', 'twitter'}
            Source of data.
        coin : str
            Name of a crypto-currency, cf augemento documentation [1]_.
        bin_size : str, {'1H', '24H'}
            Time between two observations.
        start, end : str, int or datetime
            Starting date and ending date. If string must be ISO 8601 format
            such that ('%Y-%m-%dT%H:%M:%SZ'), or if integer must be UTC
            timestamp, else can be a datetime object.

        Returns
        -------
        pd.DataFrame
            Relevant dataframe.

        References
        ----------
        .. [1] http://api-dev.augmento.ai/v0.1/documentation#introduction

        """
        if bin_size == '24H':
            nb_obs_per_day = 1
        elif bin_size == '1H':
            nb_obs_per_day = 24
        else:
            raise ValueError('Unknown bin size')

        start = intel_date(start)
        end = intel_date(end)

        dt = (end - start).days * nb_obs_per_day

        data = []

        # Iterative download
        for i in range(0, dt, 1000):
            # Request data
            data += self.get_data(
                source, coin, bin_size, start=start, end=end, start_ptr=i,
            )
            pct = i / ((dt - 1) // 1000 * 1000)
            print('Downloaded {:7.2%} [{}{}] '.format(
                pct, '=' * int(49 * pct),
                '>' * (1 - int(pct)) + ' ' * int(49 * (1 - pct))
            ), end='\r')

            # Sleep
            time.sleep(.1)

        return self._set_dataframe(data)

    def _set_dataframe(self, data):
        # Request topic names
        topics = self.send_request('topics')

        # Set dataframe
        df = pd.DataFrame(data)
        df = df.rename(columns={
            **{93: 'date', 94: 'TS'},
            **{int(k): a for k, a in topics.items()}
        })

        return df.set_index('date')


def intel_date(date, form='%Y-%m-%dT%H:%M:%SZ'):
    """ Convert date to timedate object. """
    if isinstance(date, datetime.datetime):
        return date

    elif isinstance(date, str):
        return datetime.datetime.strptime(date, form)

    elif isinstance(date, int):
        return datetime.datetime.utcfromtimestamp(date)

    else:
        raise ValueError('Unknown date object, must be datetime, string\
            (with relevent format), or int (UTC timestamp)')


if __name__ == '__main__':
    ra = RequestAugmento(logging_level='DEBUG')
