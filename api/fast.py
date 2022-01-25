from fastapi import FastAPI, File, UploadFile
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

model = load_model('../saved_models/initial_model')


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

    image = PIL.Image.open(io.BytesIO(file))
    #image = PIL.Image.open(file.file)
    image = image.resize((512, 512))
    image_arr = np.asarray(image)
    #true_index = np.argmax(y[0])
    print(image_arr)
    print(image_arr.shape)
    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = model.predict(np.expand_dims(image_arr, axis=0))
    #predicted_index = np.argmax(prediction_scores)
    #print("True label: " + label_map.values().tolist()[true_index])
    #print("Predicted label: " + class_names[predicted_index])
    print(prediction_scores)
    return {"prediction": 0}

# ValueError: embedded null byte
