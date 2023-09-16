from fastapi import FastAPI, File, UploadFile, Depends, APIRouter, Request, Response, Body, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from keras.models import load_model
from random import randint
import tensorflow as tf
import numpy as np
import pickle
import uuid
import json
import cv2

templates = Jinja2Templates(directory="pages")

obesity_router = APIRouter(
    prefix="/obesity",
    tags=["Obesity Prediction"]
)
# Loading Model here.
pickle_in = open("./obesity/obesity_ml.pkl", "rb")
obesity_ml_classifier = pickle.load(pickle_in)
obesity_cnn_classifier = load_model('./obesity/obesity_cnn.h5')
obesity_medicine_data = ""
with open('./obesity/medicine_data.json', 'r') as f: 
    obesity_medicine_data = json.load(f)

@obesity_router.post("/predict")
async def create_upload_file(
    request: Request,
    requestID: str = "TEST",
    age: int = Form(...),
    weight: float = Form(...),
    activity_level: int = Form(...),
    appetite_level: int = Form(...),
    visible_fat_deposits: int = Form(...),
    body_shape: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./obesity/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./obesity/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(obesity_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = (obesity_ml_classifier.predict([[
        age, 
        weight, 
        activity_level, 
        appetite_level, 
        visible_fat_deposits, 
        body_shape 
    ]])[0])
    prediction = ml_prediction + cnn_prediction
    if(float(prediction) > 0.85):
        # Return the prediction with medicines.
        return {"prediction": cnn_prediction, "medicine_data": obesity_medicine_data}
    else:
        return {"prediction": cnn_prediction}

@obesity_router.get("/WebUI/{username}", response_class=HTMLResponse)
async def renderHTML(request: Request, username: str):
    return templates.TemplateResponse("obesity_test.html", {"request": request, "username": username})

@obesity_router.post("/predict_WebUI", response_class=HTMLResponse)
async def response_HTML(
    request: Request,
    requestID: str = "TEST",
    age: int = Form(...),
    weight: float = Form(...),
    activity_level: int = Form(...),
    appetite_level: int = Form(...),
    visible_fat_deposits: int = Form(...),
    body_shape: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./obesity/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./obesity/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(obesity_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = (obesity_ml_classifier.predict([[
        age, 
        weight, 
        activity_level, 
        appetite_level, 
        visible_fat_deposits, 
        body_shape 
    ]])[0])
    update_ml_user_data(
        requestID,
        age, 
        weight, 
        activity_level, 
        appetite_level, 
        visible_fat_deposits, 
        body_shape,
        ml_prediction
    )
    prediction = ml_prediction + cnn_prediction
    if(float(prediction) > 0.85):
        # Return the prediction with medicines.
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": obesity_medicine_data})
    else:
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction})

def update_ml_user_data(
    requestID, 
    age, 
    weight, 
    activity_level, 
    appetite_level, 
    visible_fat_deposits, 
    body_shape,
    prediction
):
    obesity_ml_data = open("./obesity/obesity_user_data_ml.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(age) + ", " + str(weight) + ", " + str(activity_level) + ", " +  \
                str(appetite_level) + ", " + str(visible_fat_deposits) + ", " + str(body_shape) + str(prediction)
    print(new_row)
    obesity_ml_data.write('\n' + (new_row))
    obesity_ml_data.close()
    return True

def update_cnn_user_data(requestID, image_name, prediction):
    obesity_cnn_data = open("./obesity/obesity_user_data_cnn.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(image_name) + ", " + str(prediction) + ", NA" 
    print(new_row)
    obesity_cnn_data.write('\n' + (new_row))
    obesity_cnn_data.close()
    return True

@obesity_router.post("/submit_feedback")
async def submit_feedback(
    requestID: str = Form(...),
    feedback: int = Form(...)
):
    obesity_user_feedback = open("./obesity/user_feedback.txt", "a")
    new_row = str(requestID) + ", " + str(feedback) 
    print(new_row)
    obesity_user_feedback.write('\n' + (new_row))
    obesity_user_feedback.close()
    return True