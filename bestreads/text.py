"""
Contains helper functions for reducing text columns in the dataset into
workable data.
"""

import string
import warnings
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException


nltk.data.path.append('./data/nltk_data/')
nltk.download('punkt', download_dir='./data/nltk_data/', quiet=True)
nltk.download('stopwords', download_dir='./data/nltk_data/', quiet=True)


def convert_str_array(str_list):
    """
    Takes in a string of the form '1,2,3' and outputs an array with the
    integer values

    Args:
        str_list (str): String containing a set of integers

    Returns:
        id_list (numpy.ndarray): Array of integer id's
    """

    if not isinstance(str_list, str) and np.isnan(str_list):
        warnings.warn('NaN value detected in convert_str_array(str_list)',
                      category=RuntimeWarning)
        return np.array([], dtype=int)

    string_list = str_list.split(', ')
    id_list = np.array(string_list, dtype=int)

    return id_list


def get_genres(genre_and_votes, n=1):
    """
    Takes in an iterable of strings with genres and votes and returns the top n
    genres.

    Args:
        genre_and_votes (pandas.Series): Series of strings with genres and
            votes
        n (int): Number of top genres to include in result

    Returns:
        top_genres (pandas.DataFrame): DataFrame containing the top n genres
            with their votes with column names 'genre_1', 'votes_1', 'genre_2',
            etc.
    """

    column_names = [f'{stub}_{num+1}'
                    for num in range(n)
                    for stub in ['genre', 'votes']]

    top_genres = {key: [] for key in column_names}

    if type(genre_and_votes) not in (pd.Series, pd.DataFrame):
        if isinstance(genre_and_votes, str):
            genre_and_votes = [genre_and_votes]
        elif np.isnan(genre_and_votes):
            return np.nan
        else:
            raise TypeError('genre_and_votes must be a pandas.Series or '
                            + 'pandas.DataFrame object, or a string.')

    elif genre_and_votes.isnull().sum() > 0:
        warnings.warn('NaN values detected in genre_and_votes; these will be'
                      + 'skipped', category=RuntimeWarning)
        genre_and_votes = genre_and_votes[~genre_and_votes.isnull()]

    for this_str_rating in tqdm(genre_and_votes):
        split_ratings = this_str_rating.split(', ')

        # Single votes recorded as '1user' need to be changed to simply 1
        votes = [int(rating.split(' ')[-1].replace('user', ''))
                 for rating in split_ratings][:n]
        genres = [' '.join(rating.split(' ')[:-1])
                  for rating in split_ratings][:n]

        # If we ask for more genres than are available, fill in missing values
        # with np.nan
        if len(votes) < n:
            votes = votes + [np.nan, ]*(n - len(votes))
            genres = genres + [np.nan, ]*(n - len(genres))

        for x, (this_genre, this_vote) in enumerate(zip(genres, votes),
                                                    start=1):
            top_genres[f'genre_{x}'].append(this_genre)
            top_genres[f'votes_{x}'].append(this_vote)

    # If the input was a pandas object, retain original indexing
    if type(genre_and_votes) not in (pd.Series, pd.DataFrame):
        return pd.DataFrame.from_dict(top_genres)

    top_genres['index'] = genre_and_votes.index
    return pd.DataFrame.from_dict(top_genres).set_index('index')


def _clean_single_description(desc, stemmer, remove_punctuation=True):

    if not isinstance(desc, str):
        return np.nan

    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        desc = desc.translate(translator)

    tokenized = word_tokenize(desc)
    stemmed = [stemmer.stem(this_token) for this_token in tokenized]

    stop_words = set(stopwords.words('english'))
    filtered = [w for w in stemmed if not w.lower() in stop_words]

    return filtered


def clean_text(descriptions):
    """
    Stems text and removes stop words in preparation for processing.

    Args:
        descriptions (str or pandas.Series): Book description(s) to be cleaned

    Returns:
        cleaned_text(str or pandas.Series): Cleaned book descriptions. Matches
            input type.
    """

    if isinstance(descriptions, str):
        descriptions = pd.Series([descriptions])
    elif (not isinstance(descriptions, pd.Series)
          and not isinstance(descriptions, pd.DataFrame)):
        raise TypeError('descriptions must be a pandas.Series or '
                        + 'pandas.DataFrame object, or a string.')

    ps = PorterStemmer()

    return descriptions.apply(lambda desc: _clean_single_description(desc, ps))


def _is_english(text):
    """
    Returns a bool indicating whether or not text is in English.

    Args:
        text (str): The text to be recognized.

    Returns:
        english_bool (bool): Indicates if the text is English.
    """
    if not isinstance(text, str):
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def add_english_column(data):
    """
    Adds a new column to a DataFrame indicating if the 'description' column is
    in English.

    Args:
        data (pandas.DataFrame): The DataFrame constaining the descriptions of
            the books.
    """
    data['english_description'] = data['description'].apply(_is_english)
