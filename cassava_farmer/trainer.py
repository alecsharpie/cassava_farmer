from cassava_farmer.data import get_data_from_gcp, get_image_generator_gcp, get_image_generator_local
from cassava_farmer.model import build_aug_eff_model, save_model_to_gcp
from cassava_farmer.gcs import storage_upload_file, storage_upload_folder

from tensorflow.keras.callbacks import EarlyStopping

import json

from datetime import datetime

class Trainer:

    def __init__(self, where):
        self.where = where

    def train(self):

        es = EarlyStopping(patience=10)

        model = build_aug_eff_model((512, 512, 3), 5)

        if self.where == 'gcp':
            X_train, X_val, y_train, y_val = get_data_from_gcp()

            batch_size = 32

            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size

            history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size = batch_size,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=(X_val, y_val),
                            validation_steps=validation_steps,
                            callbacks = [es]).history

        elif self.where == 'gcp_fuse':
            train_ds, train_size, val_ds, val_size = get_image_generator_gcp(8)
            batch_size = 8

            steps_per_epoch = train_size // batch_size
            validation_steps = val_size // batch_size

            history = model.fit(train_ds,
                            epochs=100,
                            batch_size = batch_size,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_ds,
                            validation_steps=validation_steps,
                            callbacks = [es]).history

        elif self.where == 'local':
            train_ds, train_size, val_ds, val_size = get_image_generator_local(16)
            batch_size = 16

            steps_per_epoch = train_size // batch_size
            validation_steps = val_size // batch_size

            print('steps_per_epoch: ', steps_per_epoch)
            print('validation_steps: ', validation_steps)

            history = model.fit(
                train_ds,
                epochs=1,
                #batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[es]).history
        history_file_name = f'history/{datetime.now().strftime("history_%Y/%m/%d_%H-%M-%S")}'
        out_file = open(history_file_name, "w")
        json.dump(history, out_file, indent="")
        out_file.close()
        storage_upload_file(history_file_name)


        print(history)
        print('min accuracy', min(history['accuracy']))

        model.save('models/aug_eff_model_test')

        storage_upload_folder('models/aug_eff_model_test')


if __name__ == "__main__":
    trainer = Trainer('local')
    trainer.train()
