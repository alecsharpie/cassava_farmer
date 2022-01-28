from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from tensorflow.keras.metrics import top_k_categorical_accuracy
import functools

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

    aug_eff_model.build((None, 512, 512, 3))
    aug_eff_model.summary()
