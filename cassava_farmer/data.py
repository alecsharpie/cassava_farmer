def get_image_generator():
    #'/Users/alecsharp/me/cassava_farmer/raw_data/cassava-leaf-disease-classification/train_images'

    train_path = '../raw_data/cassava-leaf-disease-classification/train_images'

    from tensorflow.keras.utils import image_dataset_from_directory

    ds = image_dataset_from_directory(
        train_path, batch_size=32
    )
    return ds

if __name__ == "__main__":
    print(get_image_generator())
