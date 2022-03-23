import os
from tensorflow import strings
from tensorflow.image import resize
from tensorflow.io import decode_image, gfile


def get_label(file_path):
    label_map = {
        'cassava_bacterial_blight': 0,
        'cassava_brown_streak_disease': 1,
        'cassava_green_mottle': 2,
        'cassava_mosaic_disease': 3,
        'healthy': 4
    }
    # Convert the path to a the containing folder name as a string
    class_name = strings.split(file_path,
                               os.path.sep)[-2].numpy().decode("utf-8")
    # Swap out the label for its id
    return label_map[class_name]


def get_image(file_path):
    # Read in the file from google cloud storage
    jpg_file = gfile.GFile(file_path.decode('utf-8'), 'br').read()
    # Convert the jpg string to a 3D uint8 tensor
    img = decode_image(jpg_file, channels=3)
    # Resize and rescale he image to the desired size
    return resize(img, [512, 512]) / 255.0


def get_training_example(file_path):
    # get the image as a tensor from the filepath
    img = get_image(file_path.numpy())
    # get the target from the filepath
    label = get_label(file_path.numpy())
    return img, label
