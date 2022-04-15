"""
This file contains functions for downloading the Goodreads dataset for
analysis and cleaning it.
"""

from itertools import repeat
import multiprocessing
import os
import warnings
import numpy as np
import kaggle
import requests
from tqdm import tqdm


def download_dataset(savedir='data/'):
    """
    Creates a data directory if it doesn't exist and downloads the dataset
    to that directory.

    Args:
        savedir (str): The directory where the data will be saved
    """

    print(f'Downloading data csv to "{savedir}"...', end='')
    os.makedirs(savedir, exist_ok=True)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('austinreese/goodreads-books',
                                      path=savedir, unzip=True)

    print('done.')


def _download_single(single_url, single_id, savedir, verbose):
    """
    Download a single URL
    """
    file_extension = single_url.split('.')[-1]
    img_data = requests.get(single_url, stream=True)

    img_data = requests.get(single_url).content
    with open(savedir
              + f'{single_id:08}.'
              + file_extension, 'wb') as writefile:
        writefile.write(img_data)

    if verbose:
        print(f'Retrieved {single_url}')


def download_covers(url_list, id_list, savedir='data/covers/',
                    enable_multithreading=True, verbose=False):
    """
    Downloads all of the cover art images associated with each image in
    url_list and saves them in savedir.

    Args:
        url_list (iterable): An iterable of string URL's for book cover
            images
        id_list (iterable): An iterable of ID numbers.  Defaults to sequential
            numbering if not specified
        savedir (str): The directory where the images will be saved
    """

    os.makedirs(savedir, exist_ok=True)

    # Make sure id_list is iterable and set default values if not
    if not hasattr(id_list, '__iter__'):
        raise TypeError('id_list must be iterable.')
    if len(url_list) != len(id_list):
        raise ValueError('url_list and id_list must be the same length, but '
                         + f'have lengths {len(url_list)} and {len(id_list)}.')

    # If any values don't exist, raise a warning, but download the rest
    if url_list.isnull().any():
        warnings.warn('NaN value detected in url_list.  Continuing for valid '
                      + 'elements...', category=RuntimeWarning)
        id_list = id_list[~url_list.isnull()]
        url_list = url_list[~url_list.isnull()]

    if enable_multithreading:
        print(f'Downloading cover art with multithreading to "{savedir}"...')
        # Function can't be local for multiprocessing to work
        with multiprocessing.Pool() as pool:
            # tqdm is a bit janky, but it will get the job done if
            # chunksize != 1
            pool.starmap(_download_single, tqdm(list(zip(url_list, id_list,
                                                         repeat(savedir),
                                                         repeat(verbose))),
                                                total=len(url_list)),
                         chunksize=10)
    else:
        print(f'Downloading cover art to "{savedir}"...')
        for this_url, this_id in tqdm(zip(url_list, id_list),
                                      total=len(url_list)):
            download_single(this_url, this_id)


def convert_str_array(string):
    """
    Takes in a string of the form '1,2,3' and outputs an array with the
    integer values

    Args:
        string (str): String containing a set of integers
    """

    if not isinstance(string, str) and np.isnan(string):
        warnings.warn('NaN value detected in convert_str_array(string)',
                      category=RuntimeWarning)
        return np.array([], dtype=int)

    string_list = string.split(', ')
    id_list = np.array(string_list, dtype=int)

    return id_list
