from tensorflow.keras.utils import image_dataset_from_directory
from google.cloud import storage
from PIL import Image
import io
import numpy as np
import os
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import py_function
from tensorflow.data import AUTOTUNE

from cassava_farmer.data_helper import get_training_example


def get_array_sdk():

    class_names = [
        'cassava_bacterial_blight', 'cassava_brown_streak_disease',
        'cassava_green_mottle', 'cassava_mosaic_disease', 'healthy'
    ]

    client = storage.Client()

    bucket = client.get_bucket("image-datasets-alecsharpie")

    all_images = []
    all_targets = []

    for class_id, class_name in enumerate(class_names):

        blobs = bucket.list_blobs(
            prefix=f"cassava_farmer/train_images_small/{class_name}")

        images = []

        print(f"fetching {class_name} images...")
        class_count = 0
        for idx, bl in enumerate(blobs):
            # skip folder path
            if idx == 0:
                continue
            if idx % 100 == 0:
                print(f"{idx}")
            data = bl.download_as_string()
            images.append(data)
            class_count = class_count + 1

        images = [
            np.array(Image.open(io.BytesIO(img_bytes))) for img_bytes in images
        ]

        targets = [class_id] * class_count

        all_images = all_images + images

        all_targets = all_targets + targets

    X = np.array(all_images)

    y = np.array(all_targets)

    p = np.random.permutation(len(X))
    X, y = X[p], y[p]

    split_idx = int(len(X) / 0.7)

    X_train, y_train = X[:split_idx], y[:split_idx]

    X_val, y_val = X[split_idx:], y[split_idx:]

    return X_train, X_val, y_train, y_val

def get_dataset_directory(
    batch_size,
    train_path='raw_data/cassava-leaf-disease-classification/train_images_mid'
):

    train_ds = image_dataset_from_directory(
        train_path,
        batch_size=batch_size,
        subset='training',
        validation_split=.20,
        seed=42,
        image_size=(512, 512),
    )

    train_ds = train_ds.unbatch().batch(batch_size)
    train_ds = train_ds.repeat()

    val_ds = image_dataset_from_directory(train_path,
                                          batch_size=32,
                                          subset='validation',
                                          validation_split=.20,
                                          seed=42,
                                          image_size=(512, 512))

    val_ds = val_ds.unbatch().batch(batch_size)
    val_ds = val_ds.repeat()
    return train_ds, val_ds


def get_dataset_gfile(file_glob):
    # create dataset generator of file paths
    list_ds = Dataset.list_files(file_glob)
    # split the data into train and validation sets
    val_size = int(list_ds.cardinality().numpy() * 0.8)
    list_train_ds = list_ds.skip(val_size)
    list_val_ds = list_ds.take(val_size)

    # convert file paths into labelled image data
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = list_train_ds.map(lambda x: py_function(
        get_training_example, [x], [tf.float32, tf.int8]),
                                 num_parallel_calls=AUTOTUNE)
    val_ds = list_val_ds.map(lambda x: py_function(get_training_example, [x],
                                                   [tf.float32, tf.int8]),
                             num_parallel_calls=AUTOTUNE)
    return train_ds, val_ds



if __name__ == "__main__":
    print(get_dataset_mount())
