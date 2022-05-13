'''
Holds functions for building, training, saving, and reading convolutional
neural network models.
'''

import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import image
from PIL import UnidentifiedImageError
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Lambda, MaxPooling2D
from keras.models import model_from_yaml
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory


def _get_valid_data(data, min_ratings=100,
                    cover_folder='./data/covers/'):

    valid = (data[['id',
                   'cover_link',
                   'rating_count',
                   'average_rating']].notnull().all(axis=1)
             & (data['rating_count'] > min_ratings)).to_list()

    for x, row in enumerate(tqdm(data.itertuples(), total=len(data))):
        if valid[x]:
            image_fname = f'{cover_folder}{row.id:08}.jpg'
            try:
                im = image.imread(image_fname)
            except UnidentifiedImageError:
                print(image_fname + ' unreadable.')
                valid[x] = False

    valid_data = data[valid]
    valid_data.sort_values('id', inplace=True)
    return valid_data


def split_train_test_data(data, test_frac=0.2,
                          save_dir='./data/processed/cnn/',
                          cover_folder='./data/covers/'):

    valid_data = _get_valid_data(data)
    test_data = valid_data.sample(frac=test_frac)
    train_data = valid_data.drop(test_data.index)

    os.makedirs(save_dir + 'train_covers/', exist_ok=True)
    os.makedirs(save_dir + 'test_covers/', exist_ok=True)

    train_data[['id', 'average_rating']].to_csv(save_dir + 'train_ratings.csv',
                                                index=False)
    test_data[['id', 'average_rating']].to_csv(save_dir + 'test_ratings.csv',
                                               index=False)

    cover_folder = os.path.abspath(cover_folder) + '/'
    for train_id in train_data.id:
        os.symlink(cover_folder + f'{train_id:08}.jpg',
                   save_dir + f'train_covers/'
                   + f'train_{train_id:08}.jpg')

    for test_id in test_data.id:
        os.symlink(cover_folder + f'{test_id:08}.jpg',
                   save_dir + f'test_covers/'
                   + f'test_{test_id:08}.jpg')


def get_train_val_datasets_old():

    covers_train = image_dataset_from_directory('./data/processed/cnn/'
                                                + 'train_covers/',
                                                label_mode=None,
                                                batch_size=128,
                                                image_size=(500, 300),
                                                seed=1337,
                                                validation_split=0.15,
                                                subset='training')
    covers_val = image_dataset_from_directory('./data/processed/cnn/'
                                              + 'train_covers/',
                                              label_mode=None,
                                              batch_size=128,
                                              image_size=(500, 300),
                                              seed=1337,
                                              validation_split=0.15,
                                              subset='validation')

    train_ids = [int(fp[-12:-4]) for fp in covers_train.file_paths]
    val_ids = [int(fp[-12:-4]) for fp in covers_val.file_paths]

    # The second argsort here is needed to flip the reference direction
    # (we need the location where each element would be placed in a full
    # sorted set rather than the element that should be in place x)
    sorted_inds = np.argsort(np.argsort(train_ids + val_ids))
    train_inds = sorted_inds[:len(train_ids)]
    val_inds = sorted_inds[len(train_ids):]

    return (train_inds, covers_train), (val_inds, covers_val)


def build_cnn():
    '''
    Builds the CNN model

    Returns:
        model (Sequential): CNN model
    '''

    model = Sequential([
                        Conv2D(64, 3,
                               activation='relu',
                               input_shape=(500, 300, 3)),
                        MaxPooling2D(2),
                        Conv2D(128, 3, activation='relu'),
                        Conv2D(128, 3, activation='relu'),
                        MaxPooling2D(2),
                        Conv2D(256, 3, activation='relu'),
                        Conv2D(256, 3, activation='relu'),
                        MaxPooling2D(2),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dropout(0.5),
                        Dense(16, activation='relu'),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid'),
                        Lambda(lambda x: x*5)])

    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def _parse(file_name, rating):
    """Function that returns a tuple of normalized image array and rating

    Args:
        file_name (string): Path to image
        rating (float): Associated book rating (out of 5)

    returns:
        image_normalized (tf.Tensor): Normalized image tensor
        ratin (float): Associated book rating (out of 5)
    """
    # Read an image from a file
    image_string = tf.io.read_file(file_name)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [500, 300])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, rating


def create_dataset(filenames, ratings, shuffle=True, batch_size=32):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        ratings: Book ratings
        is_training: boolean to indicate training mode
    """

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

    file_paths = np.array(glob(img_dir + '*'))
    ratings = pd.read_csv(rating_path, usecols=['average_rating'],
                          squeeze=True).to_numpy()

    if len(file_paths) != len(ratings):
        raise RuntimeError(f'There are {len(file_paths)} files and '
                           + f'{len(ratings)} ratings.  Numbers of files are '
                           + 'ratings should be equal.')

    start_inds = np.arange(len(file_paths))
    np.random.shuffle(start_inds)
    train_inds = start_inds[:int((1-val_frac)*len(start_inds))]
    val_inds = start_inds[int((1-val_frac)*len(start_inds)):]

    train_files = list(file_paths[train_inds])
    train_ratings = list(ratings[train_inds])
    val_files = list(file_paths[val_inds])
    val_ratings = list(ratings[val_inds])

    return (train_files, train_ratings), (val_files, val_ratings)


def get_train_val_datasets(val_frac=0.15):

    ((train_files, train_ratings),
        (val_files, val_ratings)) = split_files_train_val(val_frac)

    train_dataset = create_dataset(train_files, train_ratings)
    val_dataset = create_dataset(val_files, val_ratings)

    return train_dataset, val_dataset


def train_cnn():

    train_dataset, val_dataset = get_train_val_datasets()

    model = build_cnn()

    history = model.fit(train_dataset, epochs=50, verbose=2,
                        validation_data=val_dataset)

    return history
