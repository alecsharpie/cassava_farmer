# home made model

from tensorflow.keras import layers
from tensorflow.keras import Sequential


def build_simple_model():

  model = Sequential([#layers.Rescaling(1./255),
                      layers.Conv2D(32, (12, 12), activation = 'relu'),
                      layers.MaxPooling2D(2),
                      layers.Conv2D(64, (6, 6), activation = 'relu'),
                      layers.MaxPooling2D(2),
                      layers.Flatten(),
                      layers.Dense(10, activation = 'relu'),
                      layers.Dropout(0.4),
                      layers.Dense(5, activation = 'softmax')
  ])

  model.compile(optimizer = Adam(learning_rate=0.001),
                loss = 'sparse_categorical_crossentropy',
                metrics = 'accuracy',
                run_eagerly=True)


  #model.build((None, 512, 512, 3))
  #model.summary()

  return model
