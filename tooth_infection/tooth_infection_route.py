# from data_class import tooth_infection_class
from fastapi import FastAPI, File, UploadFile, Depends, APIRouter
from fastapi.responses import FileResponse
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import json
import uuid
import cv2

tooth_infection_router = APIRouter(
    prefix="/tooth_infection",
    tags=["Tooth Infection Prediction"]
)
# Loading Model here.
tooth_infection_classifier = load_model('./tooth_infection/tooth_infection.h5')
tooth_infection_medicine_data = ""
with open('./tooth_infection/medicine_data.json', 'r') as f: 
    tooth_infection_medicine_data = json.load(f)
@tooth_infection_router.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./tooth_infection/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./tooth_infection/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = tooth_infection_classifier.predict(np.expand_dims(resize/255, 0))[0]
    print("CNN Prediction: ", cnn_prediction)
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    tooth_infection_user_data = open("tooth_infection/tooth_infection_user_data.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    tooth_infection_user_data.write((new_row))
    tooth_infection_user_data.close()
    if(float(cnn_prediction) > 0.5):
        # Return the prediction with medicines.
        return {"prediction": cnn_prediction, "medicine_data": tooth_infection_medicine_data}
    else:
        return {"prediction": cnn_prediction}