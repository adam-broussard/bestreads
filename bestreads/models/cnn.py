'''
Holds functions for building, training, saving, and reading convolutional
neural network models.
'''

from glob import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import image
from PIL import UnidentifiedImageError
from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Lambda, MaxPooling2D
from tensorflow.keras.optimizers import Adam  # pylint: disable=[E0611]
import tensorflow as tf


def _get_valid_data(data, min_ratings=100,
                    cover_folder='./data/covers/'):
    '''
    Eliminates any data with a low nubmer of ratings, null values for
    average_rating, or unreadable cover images

    Args:
        data (pandas.DataFrame): All of the goodreads data (or at least a
            subset with 'id', 'cover_link', 'rating_count', and
            'average_rating' defined)
        min_ratings (int): The minimum number of ratings for a book to not be
            excluded
        cover_folder (string): The file path where covers are saved

    Returns:
        valid_data (pd.DataFrame): The subset of data that has valid entries
            and cover images
    '''

    # Maks sure all of the necessary data is present
    valid = (data[['id',
                   'cover_link',
                   'rating_count',
                   'average_rating']].notnull().all(axis=1)
             & (data['rating_count'] > min_ratings)).to_list()

    # Make sure the cover art is readable (which is generally true, but has a
    # few exceptions)
    for x, row in enumerate(tqdm(data.itertuples(), total=len(data))):
        if valid[x]:
            image_fname = f'{cover_folder}{row.id:08}.jpg'
            try:
                image.imread(image_fname)
            except UnidentifiedImageError:
                print(image_fname + ' unreadable.')
                valid[x] = False

    valid_data = data[valid]

    # It is important to sort because later functions rely on being able to
    # associate cover images with ratings using the sorted filepaths (which
    # are id numbers)
    valid_data.sort_values('id', inplace=True)
    return valid_data


def split_train_test_data(data, test_frac=0.2,
                          save_dir='./data/processed/cnn/',
                          cover_dir='./data/covers/'):
    '''
    Generates folders for train and test set cover images and generates
    symlinks pointing to the original files to save space.

    Args:
        data (pd.DataFrame): All of the goodreads data
        test_frac (float): The fraction of the data to be split into a test set
        save_dir (string): The directory in which to save the train and test
            set covers
        cover_dir (string): The directory containing the cover image files
    '''

    valid_data = _get_valid_data(data)

    # Split into test and train
    test_data = valid_data.sample(frac=test_frac)
    train_data = valid_data.drop(test_data.index)

    os.makedirs(save_dir + 'train_covers/', exist_ok=True)
    os.makedirs(save_dir + 'test_covers/', exist_ok=True)

    # Save test and train datasets (note only the average_rating is truly
    # necessary, but including the id allows for some checking that things
    # match up again properly later if needed)
    train_data[['id', 'average_rating']].to_csv(save_dir + 'train_ratings.csv',
                                                index=False)
    test_data[['id', 'average_rating']].to_csv(save_dir + 'test_ratings.csv',
                                               index=False)

    # Generate folders of symlinks for training and testing datasets
    cover_dir = os.path.abspath(cover_dir) + '/'
    for train_id in train_data.id:
        os.symlink(cover_dir + f'{train_id:08}.jpg',
                   save_dir + 'train_covers/'
                   + f'train_{train_id:08}.jpg')

    for test_id in test_data.id:
        os.symlink(cover_dir + f'{test_id:08}.jpg',
                   save_dir + 'test_covers/'
                   + f'test_{test_id:08}.jpg')


def build_cnn():
    '''
    Builds the CNN model

    Returns:
        model (Sequential): CNN model
    '''

    # This setup with the defualt batch size uses about 7.5 GB of memory and
    # takes around 30 minutes per epoch of training on Adam's laptop.
    model = Sequential([
                        Conv2D(64, 3,
                               activation='relu',
                               input_shape=(450, 300, 3)),
                        MaxPooling2D(4),
                        Conv2D(128, 3, activation='relu'),
                        Conv2D(128, 3, activation='relu'),
                        MaxPooling2D(2),
                        Conv2D(64, 3, activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dropout(0.5),
                        Dense(16, activation='relu'),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid'),
                        Lambda(lambda x: x*5)])

    # Note we could divide the targets by 5 instead of multiplying the output
    # by 5, but I don't think they're functionally very different.

    model.compile(loss='mse',
                  optimizer=Adam())
    return model


def _parse(file_name, rating):
    '''
    Function that returns a tuple of normalized image array and rating

    Args:
        file_name (string): Path to image
        rating (float): Associated book rating (out of 5)

    returns:
        image_normalized (tf.Tensor): Normalized image tensor
        ratin (float): Associated book rating (out of 5)
    '''
    # Read an image from a file
    image_string = tf.io.read_file(file_name)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [450, 300])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, rating


def create_dataset(filenames, ratings, shuffle=False, batch_size=32):
    '''
    Create a tensorflow dataset object and return it.

    Args:
        filenames (iter): List of image paths
        ratings (iter): List of associated book ratings
        shuffle (bool): Whether or not to shuffle the dataset after generating
            it (note this is less effective than shuffling the filenames and
            ratings beforehand instead because the whole dataset cannot be
            stored in memory simultaneously.)
        batch_size (int): The number of images per batch

    Returns:
        dataset (tf.data.Dataset): A dataset containing the image and rating
            data
    '''

    # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU
    # idle time
    autotune = tf.data.experimental.AUTOTUNE

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, ratings))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(_parse, num_parallel_calls=autotune)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)

    # Batch the data for multiple steps
    dataset = dataset.batch(batch_size)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset


def split_files_train_val(val_frac=0.15,
                          rating_path='./data/processed/cnn/train_ratings.csv',
                          img_dir='./data/processed/cnn/train_covers/'):
    '''
    Splits the training set into a true training set and a validation set using
    the filenames and ratings.

    Args:
        val_frac (float): The fraction of the original training set to use for
            validation
        rating_path (string): The path to the .csv file containing an
            'average_ratings' column.  Should correspond to the sorted cover
            image filenames.
        img_dir (string): The directory containing cover images for training

    Returns:
        train_files (iter): Shuffled list of cover filenames for training
        train_ratings (iter): Shuffled list of ratings for training
        val_files (iter): Shuffled list of cover filenames for validation
        val_ratings (iter): Shuffled list of ratings for validation
    '''

    # Make sure to sort!  That is how we correlate these with their ratings.
    file_paths = np.array(sorted(glob(img_dir + '*')))
    ratings = pd.read_csv(rating_path, usecols=['average_rating'],
                          squeeze=True).to_numpy()

    if len(file_paths) != len(ratings):
        raise RuntimeError(f'There are {len(file_paths)} files and '
                           + f'{len(ratings)} ratings.  Numbers of files are '
                           + 'ratings should be equal.')

    # Split into training and validation sets and split the filenames and
    # ratings accordingly
    start_inds = np.arange(len(file_paths))
    np.random.shuffle(start_inds)
    train_inds = start_inds[:int((1-val_frac)*len(start_inds))]
    val_inds = start_inds[int((1-val_frac)*len(start_inds)):]

    train_files = list(file_paths[train_inds])
    train_ratings = list(ratings[train_inds])
    val_files = list(file_paths[val_inds])
    val_ratings = list(ratings[val_inds])

    return (train_files, train_ratings), (val_files, val_ratings)


def get_train_val_datasets(batch_size = 32, val_frac=0.15):
    '''
    Splits data into training and validation sets and generates Dataset objects
    to hold them.

    Args:
        val_frac (float): Fraction of the training data to reserve for
            validation

    Returns:
        train_dataset (tf.data.Dataset): The batched training data with ratings
        val_dataset (tf.data.Dataset): the batched validation data with ratings
    '''

    ((train_files, train_ratings),
        (val_files, val_ratings)) = split_files_train_val(val_frac)

    train_dataset = create_dataset(train_files, train_ratings, batch_size = batch_size)
    val_dataset = create_dataset(val_files, val_ratings, batch_size = batch_size)

    return train_dataset, val_dataset


def train_cnn(epochs=50):
    '''
    Get the training and validation datasets, and then train the CNN.

    Args:
        epochs (int): The number of times to loop through the training data

    Returns:
        history (tf.keras.callbacks.History): An object holding the training
            history of the model
        model (tf.keras.models.Sequential): The CNN model
    '''

    train_dataset, val_dataset = get_train_val_datasets()

    model = build_cnn()

    history = model.fit(train_dataset, epochs=epochs, verbose=1,
                        validation_data=val_dataset)

    return history, model


def save_model(model, fname='cnn', save_dir='./'):
    '''
    Save CNN model to file

    Args:
        model (tf.keras.Model): The model to save
        fname (string): The name of the files
        save_dir (string): The directory in which to save the files
    '''

    model_yaml = model.to_yaml()
    yamlfile = save_dir + fname + '.yaml'
    h5file = save_dir + fname + '.h5'

    # Save model structure
    with open(yamlfile, 'w', encoding='ascii') as yamlwrite:
        yamlwrite.write(model_yaml)

    # Serialize weights to HDF5
    model.save_weights(h5file)


def read_model(fname='cnn', read_dir='./'):
    '''
    Read the CNN model from file

    Args:
        fname (string): The name of the files
        read_dir (string): The directory in which to read the saved model

    Returns:
        loaded_model (tf.keras.Model): The model read from file
    '''

    yamlfile = read_dir + fname + '.yaml'
    h5file = read_dir + fname + '.h5'

    with open(yamlfile, 'r', encoding='ascii') as readyaml:
        loaded_model_yaml = readyaml.read()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights(h5file)
    loaded_model.compile(loss='mse', optimizer=Adam())

    return loaded_model
