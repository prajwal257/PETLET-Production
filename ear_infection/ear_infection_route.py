# from data_class import ear_infection_class
from fastapi import FastAPI, File, UploadFile, Depends, APIRouter
from fastapi.responses import FileResponse
from data_class import feedback_class
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import json
import uuid
import cv2

ear_infection_router = APIRouter(
    prefix="/ear_infection",
    tags=["Ear Infection Prediction"]
)
# Loading Model here.
earinfection_classifier = load_model('./ear_infection/ear_infection.h5')
ear_infection_medicine_data = ""
with open('./ear_infection/medicine_data.json', 'r') as f: 
    ear_infection_medicine_data = json.load(f)
@ear_infection_router.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Predicting from CNN.
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./ear_infection/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./ear_infection/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = earinfection_classifier.predict(np.expand_dims(resize/255, 0))[0]
    print("CNN Prediction: ", cnn_prediction)
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    earinfection_data = open("ear_infection/ear_infection_user_data.txt", "a")
    new_row = str(file.filename) + ", " + str(cnn_prediction) + ", NA \n" 
    print(new_row)
    earinfection_data.write((new_row))
    earinfection_data.close()
    if(float(cnn_prediction) > 0.5):
        # Return the prediction with medicines.
        return {"prediction": cnn_prediction, "medicine_data": ear_infection_medicine_data}
    else:
        return {"prediction": cnn_prediction}
    
@ear_infection_router.post("/feedback")
async def submit_feedback(data: feedback_class):
    ear_infection_feedback   = open("./ear_infection/feedback_user_data.txt", "a")
    requestID = str(data.requestID)
    feedback_score = str(data.feedback_score)
    new_row = requestID + ", " + feedback_score + "\n"
    print(new_row)
    ear_infection_feedback.write(new_row)
    return True