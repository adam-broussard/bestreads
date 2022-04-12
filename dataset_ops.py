"""
This file contains functions for downloading the Goodreads dataset for
analysis and cleaning it.
"""

import os

import warnings
import numpy as np
import kaggle


def download_dataset(savedir='data/'):
    """
    Creates a data directory if it doesn't exist and downloads the dataset
    to that directory.

    Args:
        savedir (str): The directory where the data will be saved
    """


    print('Downloading data csv...', end='')
    os.makedirs(savedir, exist_ok=True)


    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('austinreese/goodreads-books',
                                      path=savedir, unzip=True)

    print('done.')

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

