"""
This file contains functions for downloading the Goodreads dataset for
analysis and cleaning it.
"""

from itertools import repeat
import json
import multiprocessing
import random
import os
import warnings
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
            _download_single(this_url, this_id, savedir, verbose)


def subsample_json(src_file_path, dest_file_path,
                   min_ratings=100, samples=100000, overshoot=3.):
    """
    Randomly samples book entries from a JSON file to produce a subsample such
    that each book has at least 100 associated ratings.  Saves the results into
    a new JSON.

    Args:
        src_file_path (string): Path to the file to be read.
        dest_file_path (string): Where to save the new JSON file with the
            subsample.
        min_ratings (int): Minimum number of ratings allowed for books in the
            subsample.
        samples (int): Number of books to draw for the subsample.
        overshoot (float): Determines how large chunks of books should be that
             are considered for the subsample.  Larger numbers will use more
             memory, but will also converge faster.
    """

    data = []

    with open(src_file_path, 'r', encoding='utf-8') as rf:

        # Count the lines
        num_lines = len([None for _ in rf])
        line_read_order = list(range(num_lines))

    # Create a random line ordering
    random.seed(3423)
    random.shuffle(line_read_order)

    ind = 0
    while len(data) < samples and ind < num_lines-1:

        num_lines_to_read = max([int((samples - len(data))*overshoot), 10000])

        lines_to_read = set(line_read_order[ind:ind+num_lines_to_read])
        readlines = amnestic_reader(src_file_path, lines_to_read)

        for line in readlines:
            info = json.loads(line)

            try:
                if (int(info['ratings_count']) >= min_ratings
                   and info['image_url'] != ''):
                    data.append(info)
            except ValueError:
                pass

        print(f'{len(data)} of {samples} drawn.')

        ind += num_lines_to_read

    if len(data) > samples:
        print(f'Paring down samples to {samples}.')
        data = data[:samples]

    # Save the new sample to file
    with open(dest_file_path, 'w', encoding='utf-8') as wf:
        for info in data:
            wf.write(json.dumps(info))
            wf.write('\n')


def amnestic_reader(file_path, line_nums):
    """
    A "forgetful" reader that reads specific lines from a file and forgets
    everything else.  This is particularly handy when reading lines from
    very large files that can't be stored in memory.

    Args:
        file_path (string): Path to file that will be read in
        line_nums (string): An iterable of line numbers to read

    Yields:
        line (string): Line text
    """

    with open(file_path, 'r', encoding='utf-8') as rf:
        for line_num, line in enumerate(rf):
            if line_num in line_nums:
                yield line


def get_img_info_json(file_path):
    """
    Gets image URL's from a JSON file containing book information.

    Args:
        file_path (string): Path to file to be read

    Returns:
        bookdata (dict): Contains book ID numbers and cover image URL's
    """

    bookdata = {'image_url': [], 'book_id': []}

    with open(file_path, 'r', encoding='utf-8') as rf:

        for line in rf:
            linedata = json.loads(line)

            bookdata['book_id'].append(linedata['book_id'])
            bookdata['image_url'].append(linedata['image_url'])

    return bookdata
