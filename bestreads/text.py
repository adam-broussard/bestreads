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

    for this_str_rating in tqdm(genre_and_votes):
        if isinstance(this_str_rating, float):
            votes = []
            genres = []

        else:

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
    in English. The new DataFrame is returned as a copy.

    Args:
        data (pandas.DataFrame): The DataFrame containing the descriptions of
            the books.
    """
    return data.assign(
        english_description=data['description'].apply(_is_english)
    )


def combine_genres(genres, descriptions):
    """
    Takes in the genres and cleaned, tokenized, stemmed descriptions to return
    the combined processed descriptions associated with each genre.

    Args:
        genres (pandas.Series or pandas.DataFrame): Each book's genre as a
            string
        descriptions (pandas.Series or pandas.DataFrame): Each book's
            description (as a list of strings where each element is a
            tokenized, stemmed word)

    Returns:
        combined (dict): A dictionary where each key is a genre and its value
            is a combined pandas.Series of all description words
    """

    # Make a dict where keys are unique genres

    unique_genres = set(genres)
    # Get rid of nans if they're there, but don't complain otherwise
    unique_genres.discard(np.nan)
    combined = {key: [] for key in unique_genres}

    for key in combined.keys():
        genre_descriptions = descriptions[genres == key]
        for single_desc in genre_descriptions:
            if not isinstance(single_desc, float):  # Get rid of lingering nans
                combined[key] += single_desc

    # Turn into series for easy counting later
    for key in combined.keys():
        combined[key] = pd.Series(combined[key])

    return combined


def tf_idf(combined):
    """
    Takes in a combined dictionary of description words grouped by genre and
    runs tf-idf on each unique word.

    Args:
        combined (dict): A dictionary where each key is a genre and each value
            is the combined descriptions for all books in that genre.

    Returns:
        result (pandas.DataFrame): Each element is the index word's term
            frequency-inverse document frequency value within descriptions for
            the corresponding genre in the column name.  Missing words are
            represented with Nans.
    """

    unique_word_counts = {}
    for genre, description in combined.items():
        unique_word_counts[genre] = description.value_counts(sort=False)

    result = {key: {} for key in combined.keys()}
    num_docs = len(combined)
    for genre, word_counts in unique_word_counts.items():
        num_words = word_counts.sum()
        for word, count in word_counts.items():
            tf = count/num_words
            num_docs_contain = sum([word in unique_word_counts[gen].index
                                    for gen in combined.keys()
                                    if gen != genre])
            idf = np.log(num_docs/(1+num_docs_contain))
            result[genre][word] = tf*idf

    return pd.DataFrame.from_dict(result)
