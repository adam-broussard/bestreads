'''
Holds functions for building, training, saving, and reading convolutional
neural network models.
'''

import os
import pandas as pd
from tqdm import tqdm
from matplotlib import image
from PIL import UnidentifiedImageError
# pylint: disable=[E0611,E0401]
from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# pylint: enable-[E0611,E0401]


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


def split_train_val_test_data(data, test_frac=0.2, val_frac=0.2,
                              save_dir='./data/processed/cnn/',
                              cover_dir='./data/covers/'):
    '''
    Generates folders for train and test set cover images and generates
    symlinks pointing to the original files to save space.

    Args:
        data (pd.DataFrame): All of the goodreads data
        test_frac (float): The fraction of the data to be split into a test set
        val_frac (float): The fraction of the data to be split into a
            validation set
        save_dir (string): The directory in which to save the train and test
            set covers
        cover_dir (string): The directory containing the cover image files
    '''

    valid_data = _get_valid_data(data)

    # Split into test and train
    test_data = valid_data.sample(frac=test_frac, random_state=420)
    train_data = valid_data.drop(test_data.index)

    val_data = train_data.sample(frac=val_frac, random_state=421)
    train_data = train_data.drop(val_data.index)

    os.makedirs(save_dir + 'train_covers/', exist_ok=True)
    os.makedirs(save_dir + 'test_covers/', exist_ok=True)
    os.makedirs(save_dir + 'val_covers/', exist_ok=True)

    # Generate file_path column

    train_data['file_path'] = [save_dir + f'train_covers/train_{idnum:08}.jpg'
                               for idnum in train_data['id']]
    test_data['file_path'] = [save_dir + f'test_covers/test_{idnum:08}.jpg'
                              for idnum in test_data['id']]
    val_data['file_path'] = [save_dir + f'val_covers/val_{idnum:08}.jpg'
                             for idnum in val_data['id']]

    # Save test and train datasets (note only the average_rating is truly
    # necessary, but including the id allows for some checking that things
    # match up again properly later if needed)

    cover_dir = os.path.abspath(cover_dir) + '/'
    for sample, name in zip([train_data, test_data, val_data],
                            ['train', 'test', 'val']):
        save_cols = ['file_path', 'average_rating']
        sample[save_cols].to_csv(save_dir + name + '_ratings.csv',
                                 index=False)

        # Generate folders of symlinks for training and testing datasets
        for idnum in sample.id:
            os.symlink(cover_dir + f'{idnum:08}.jpg',
                       save_dir + name + '_covers/'
                       + name + f'_{idnum:08}.jpg')


def build_cnn():
    '''
    Builds the CNN model

    Returns:
        model (Sequential): CNN model
    '''

    model = Sequential([
                        Conv2D(64, 3,
                               activation='relu',
                               input_shape=(450, 300, 3)),
                        MaxPooling2D(4),
                        Conv2D(64, 3, activation='relu'),
                        MaxPooling2D(4),
                        Conv2D(32, 3, activation='relu'),
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


def train_cnn(epochs=25, batch_size=32,
              train_data='./data/processed/cnn/train_ratings.csv',
              val_data='./data/processed/cnn/train_ratings.csv'):
    '''
    Get the training and validation datasets, and then train the CNN.

    Args:
        epochs (int): The number of times to loop through the training data

    Returns:
        history (tf.keras.callbacks.History): An object holding the training
            history of the model
        model (tf.keras.models.Sequential): The CNN model
    '''

    batch_size = 32

    train_data = pd.read_csv('./data/processed/cnn/train_ratings.csv')
    val_data = pd.read_csv('./data/processed/cnn/val_ratings.csv')

    # Shuffle the data
    train_data = train_data.sample(frac=1, random_state=2974)
    val_data = val_data.sample(frac=1, random_state=2143)

    # Create Tensorflow datasets
    train_dataset = cnn.create_dataset(train_data.file_path,
                                       train_data.average_rating,
                                       batch_size=batch_size)
    val_dataset = cnn.create_dataset(val_data.file_path,
                                     val_data.average_rating,
                                     batch_size=batch_size)

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


def read_model(fname='cnn', read_dir='./models/'):
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
