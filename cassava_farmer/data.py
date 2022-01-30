from tensorflow.keras.utils import image_dataset_from_directory
from google.cloud import storage
from PIL import Image
import io
import numpy as np

def get_data_from_gcp():

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

    return X_train, X_val, y_val, y_train


def get_image_generator_local(batch_size):

    train_path = 'raw_data/cassava-leaf-disease-classification/train_images'

    train_ds = image_dataset_from_directory(
        train_path, batch_size=32, subset='training', validation_split=.20, seed = 42, image_size=(512, 512),
    )

    class_names = train_ds.class_names

    train_size = train_ds.cardinality().numpy()
    train_ds = train_ds.unbatch().batch(batch_size)
    train_ds = train_ds.repeat()


    val_ds = image_dataset_from_directory(
        train_path, batch_size=32, subset='validation', validation_split=.20, seed = 42, image_size=(512, 512)
    )

    val_size = val_ds.cardinality().numpy()
    val_ds = val_ds.unbatch().batch(batch_size)
    val_ds = val_ds.repeat()
    return train_ds, train_size, val_ds, val_size

if __name__ == "__main__":
    print(get_image_generator_local())
