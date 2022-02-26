import os

from google.cloud import storage
from cassava_farmer.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION

# files = ['my_model/assets/vocab.txt',
# 'my_model/variables/variables.data-00000-of-00001',
# 'my_model/variables/variables.index',
# 'my_model/keras_metadata.pb',
# 'my_model/saved_model.pb']


def storage_upload_folder(folder_to_upload='my_model',
                          gcp_folder='models'):
    local_file_paths = []
    for parent, dirnames, filenames in os.walk(folder_to_upload):
        for filename in filenames:
            local_file_paths.append(parent + '/' + filename)
    client = storage.Client().bucket(BUCKET_NAME)
    for file_name in local_file_paths:
        cloud_storage_location = f"/{MODEL_NAME}/{MODEL_VERSION}/{file_name}"
        blob = client.blob(cloud_storage_location)
        blob.upload_from_filename(file_name)
        print('folder uploaded sucessfully')


def storage_upload_file(rm=False,
                        file_name='my_model_history.json',
                        gcp_folder='history'):
    client = storage.Client().bucket(BUCKET_NAME)
    local_model_name = f'{gcp_folder}/{file_name}'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(file_name)
    print("file uploaded successfully")
    if rm:
        os.remove(file_name)
