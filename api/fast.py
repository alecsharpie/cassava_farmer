from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
import numpy as np
import PIL

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
async def create_file(file: bytes = File(...)):
    model = load_model('../saved_models/initial_model')

    image = PIL.Image.open(file)
    image = image.resize((512, 512))
    #true_index = np.argmax(y[0])

    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    #predicted_index = np.argmax(prediction_scores)
    #print("True label: " + label_map.values().tolist()[true_index])
    #print("Predicted label: " + class_names[predicted_index])
    return {"prediction": prediction_scores}
