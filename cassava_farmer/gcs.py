import os

from google.cloud import storage
from cassava_farmer.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION

# files = ['my_model/assets/vocab.txt',
# 'my_model/variables/variables.data-00000-of-00001',
# 'my_model/variables/variables.index',
# 'my_model/keras_metadata.pb',
# 'my_model/saved_model.pb']


def storage_upload_folder(path_to_folder='my_model'):
    local_file_paths = []
    for parent, dirnames, filenames in os.walk(path_to_folder):
        for filename in filenames:
            local_file_paths.append(parent + '/' + filename)
    client = storage.Client().bucket(BUCKET_NAME)
    for file_name in local_file_paths:
        cloud_storage_location = f"{MODEL_NAME}/{MODEL_VERSION}/{file_name}"
        blob = client.blob(cloud_storage_location)
        blob.upload_from_filename(file_name)
    print('folder uploaded sucessfully')


def storage_upload_file(path_to_file='history/my_model_history.json',
                        gcp_folder='history'):
    client = storage.Client().bucket(BUCKET_NAME)
    file_name = path_to_file.split('/')[-1]
    cloud_storage_location = f"{MODEL_NAME}/{MODEL_VERSION}/{file_name}"
    blob = client.blob(cloud_storage_location)
    blob.upload_from_filename(path_to_file)
    print("file uploaded successfully")
