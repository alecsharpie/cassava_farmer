from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
import numpy as np
import PIL
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

label_map = {
    '0': 'cassava_bacterial_blight',
    '1': 'cassava_brown_streak_disease',
    '2': 'cassava_green_mottle',
    '3': 'cassava_mosaic_disease',
    '4': 'healthy'
}

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(file: bytes = File(...)):

    model = load_model('saved_models/initial_model2')

    image = PIL.Image.open(io.BytesIO(file))
    image = image.resize((512, 512))

    image_arr = np.asarray(image) / 255.0
    print('input size: ', image_arr.shape)

    prediction_scores = model.predict(np.expand_dims(image_arr, axis=0))

    class_names = list(label_map.values())
    predictions_dict = {
        key: float(value)
        for key, value in zip(class_names, list(prediction_scores[0]))
    }

    return predictions_dict
