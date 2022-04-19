"""
Contains helper functions for reducing text columns in the dataset into
workable data.
"""
from langdetect import detect, LangDetectException

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
        data (pandas.DataFrame): The DataFrame constaining the descriptions of the books.
    """
    data['english_description'] = data['description'].apply(_is_english)
