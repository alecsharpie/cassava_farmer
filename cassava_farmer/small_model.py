# home made model

from tensorflow.keras import layers
from tensorflow.keras import Sequential


def build_model():

  augmentation = Sequential([
          layers.RandomContrast(0.2),
          layers.RandomRotation(40),
          layers.RandomTranslation(0, 0.2),
          layers.RandomTranslation(0.2, 0),
          layers.RandomZoom(0.2, 0.2),
          layers.RandomFlip(mode="horizontal")
      ])


  model = Sequential([layers.Rescaling(1./255),
                      #layers.Resizing(256, 256),
                      #augmentation,
                      layers.Conv2D(16, (24, 24), activation = 'relu'),
                      layers.MaxPooling2D(2),
                      layers.Conv2D(32, (12, 12), activation = 'relu'),
                      layers.MaxPooling2D(2),
                      layers.Conv2D(64, (6, 6), activation = 'relu'),
                      layers.MaxPooling2D(2),
                      layers.Conv2D(128, (5, 5), activation = 'relu'),
                      layers.MaxPooling2D(2),
                      layers.Flatten(),
                      layers.Dense(100, activation = 'relu'),
                      layers.Dropout(0.3),
                      layers.Dense(10, activation = 'relu'),
                      layers.Dropout(0.4),
                      layers.Dense(5, activation = 'softmax')
  ])

  model.compile(optimizer = Adam(learning_rate=0.0000001),
                loss = 'sparse_categorical_crossentropy',
                metrics = 'accuracy',
                run_eagerly=True)


  model.build((None, 512, 512, 3))
  model.summary()

  return model
