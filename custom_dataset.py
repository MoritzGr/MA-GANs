import os

import numpy
import pickle
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

import settings


def get_wikiart_dataset_iterator():
    ds_train = tf.keras.utils.image_dataset_from_directory(
        settings.ORIGINAL_IMAGES_PATH,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=1,
        image_size=(settings.IMG_SIZE, settings.IMG_SIZE),
        crop_to_aspect_ratio=True
    )
    return ds_train.__iter__()


def iterator_to_datasetarray(iterator):
    images = []
    labels = []
    while True:
        try:
            sample = iterator.__next__()
            image, label = sample
            image = image.numpy()
            label = label.numpy()
            images.append(image[0])
            labels.append(label[0])
        except InvalidArgumentError:
            print('skipping corrupt file')
        except StopIteration:
            break

    output = [numpy.asarray(images), numpy.asarray(labels)]
    return output


def get_wikiart_dataset():
    filepath = settings.RESIZED_IMAGES_PATH + "Wikiart_" + str(settings.IMG_SIZE) + ".pkl"
    if os.path.isfile(filepath):
        with open(filepath, "rb") as file:
            output = pickle.load(file)
    else:
        if not os.path.exists(os.path.dirname(filepath)):
            os.mkdir(settings.RESIZED_IMAGES_PATH)

        output = iterator_to_datasetarray(get_wikiart_dataset_iterator())

        with open(filepath, "wb") as file:
            pickle.dump(output, file, pickle.HIGHEST_PROTOCOL)
    return output


def get_emoji_dataset_iterator(by_category, faces_only):
    filepath = settings.EMOJI_IMAGES_PATH

    if by_category:
        filepath = filepath + "/by_category"
    else:
        filepath = filepath + "/by_manufacturer"

    if faces_only:
        filepath = filepath + "_faces_only/"
    else:
        filepath = filepath + "/"

    ds_train = tf.keras.utils.image_dataset_from_directory(
        filepath,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=1,
        image_size=(settings.IMG_SIZE, settings.IMG_SIZE),
        crop_to_aspect_ratio=True
    )
    return ds_train.__iter__()


def get_emoji_dataset(by_category=False, faces_only=True):
    filepath = settings.RESIZED_IMAGES_PATH + "Emoji_"
    if by_category:
        filepath = filepath + "bycat_"
    else:
        filepath = filepath + "byman_"

    if faces_only:
        filepath = filepath + "facesonly_"

    filepath = filepath + str(settings.IMG_SIZE) + ".pkl"

    if os.path.isfile(filepath):
        with open(filepath, "rb") as file:
            output = pickle.load(file)
    else:
        if not os.path.exists(os.path.dirname(filepath)):
            os.mkdir(settings.RESIZED_IMAGES_PATH)

        output = iterator_to_datasetarray(get_emoji_dataset_iterator(by_category, faces_only))

        with open(filepath, "wb") as file:
            pickle.dump(output, file, pickle.HIGHEST_PROTOCOL)
    return output
