from cassava_farmer.data import get_data_from_gcp, get_image_generator_local
from cassava_farmer.model import build_aug_eff_model


class Trainer:

    def __init__(self, where):
        self.where = where

    def train(self):

        model = build_aug_eff_model()

        if self.where == 'gcp':
            X_train, X_val, y_val, y_train = get_data_from_gcp()

            batch_size = 8

            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size

            history = model.fit(X_train, y_train,
                            epochs=1,
                            batch_size = batch_size,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=(X_val, y_val),
                            validation_steps=validation_steps).history


        elif self.where == 'local':
            train_ds, val_ds = get_image_generator_local(8)
            batch_size = 8

            steps_per_epoch = train_ds.cardinality().numpy() // batch_size
            validation_steps = val_ds.cardinality().numpy() // batch_size

            history = model.fit(train_ds,
                            epochs=1,
                            batch_size = batch_size,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_ds,
                            validation_steps=validation_steps).history


        print('min accuracy', min(history['accuracy']))


if __name__ == "__main__":
    trainer = Trainer('local')
    trainer.train()
