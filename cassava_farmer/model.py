from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from tensorflow.config import run_functions_eagerly

from google.cloud import storage

from tensorflow.keras.metrics import top_k_categorical_accuracy
import functools

#run_functions_eagerly(True)

def build_aug_eff_model(input_shape, output_classes):

    augmentation = Sequential([
        layers.RandomContrast(0.2),
        layers.RandomRotation(40),
        layers.RandomTranslation(0, 0.2),
        layers.RandomTranslation(0.2, 0),
        layers.RandomZoom(0.2, 0.2),
        layers.RandomFlip(mode="horizontal")
    ])

    dummy_input = layers.Input(shape=input_shape)

    topless_efficient_net = EfficientNetB0(include_top=False,
                                        weights='imagenet',
                                        input_tensor=dummy_input,
                                        pooling='max')

    aug_eff_model = Sequential([
        layers.Resizing(512, 512),
        augmentation,
        topless_efficient_net,
        layers.Dropout(0.2),
        layers.Dense(output_classes, activation='softmax')
    ])

    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'

    aug_eff_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy', top3_acc])
                        #run_eagerly=True)

    aug_eff_model.build((None, 512, 512, 3))
    aug_eff_model.summary()

    aug_eff_model.layers[2].trainable = False

    return aug_eff_model


def save_model_to_gcp():

    BUCKET_NAME = "image-datasets-alecsharpie"
    model_storage_location = "cassava_farmer/models/aug_eff_model.h5"
    history_storage_location = "cassava_farmer/models/history_aug_eff_model.json"

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(model_storage_location)
    blob.upload_from_filename('aug_eff_model.h5')

    blob = bucket.blob(history_storage_location)
    blob.upload_from_filename('history_aug_eff_model.json')



def get_model_from_gcp_blob():

    # get data from my google storage bucket
    BUCKET_NAME = "image-datasets-alecsharpie"

    cloud_storage_location = "cassava_farmer/models/"

    local_model_filename = "trained_aug_eff_model"

    client = storage.Client()  # verifies $GOOGLE_APPLICATION_CREDENTIALS

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(cloud_storage_location)

    blob.download_to_filename(local_model_filename)
    print('Model Downloaded!')
    return None
