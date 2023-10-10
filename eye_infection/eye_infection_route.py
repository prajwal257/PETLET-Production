from fastapi import FastAPI, File, UploadFile, Depends, APIRouter, Request, Response, Body, Form
from fastapi.templating import Jinja2Templates
from authorized_users import authorized_users
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from data_class import eye_infection_class, feedback_class
from keras.models import load_model
from random import randint
import tensorflow as tf
import numpy as np
import pickle
import uuid
import json
import cv2

templates = Jinja2Templates(directory="pages")

eye_infection_router = APIRouter(
    prefix="/eye_infection",
    tags=["Eye Infection Prediction"]
)

# Loading Model here.
eye_infection_cnn_classifier = load_model('./eye_infection/eye_infection_cnn.h5')
pickle_in = open("./eye_infection/eye_infection_ml.pkl", "rb")
eye_infection_ml_classifier = pickle.load(pickle_in)

# Loading the medicine data.
eye_infection_medicine_data = ""
with open('./eye_infection/medicine_data.json', 'r') as f: 
    eye_infection_medicine_data = json.load(f)

@eye_infection_router.post('/predict')
async def eye_infection_classifier(
    # Enter all the paramaters with their datatypes...
    request: Request,
    requestID: str = "TEST",
    age: int = Form(...),
    breed: int = Form(...),
    sex: int = Form(...),
    redness: int = Form(...),
    swelling: int = Form(...),
    discharge: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./eye_infection/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./eye_infection/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(eye_infection_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = (eye_infection_ml_classifier.predict([[age, breed, sex, redness, swelling, discharge]])[0])
    update_ml_user_data(requestID, age, breed, sex, redness, swelling, discharge, ml_prediction)
    prediction = float(ml_prediction + cnn_prediction)
    if(prediction > 0.85):
        return {"prediction": prediction, "medicine_data": eye_infection_medicine_data}
        # return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": eye_infection_medicine_data})
    else:
        return {"prediction": prediction}
        # return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": eye_infection_medicine_data})

@eye_infection_router.get("/WebUI/{username}", response_class=HTMLResponse)
async def render_HTML(request: Request, username: str):
    if(username in authorized_users):
        return templates.TemplateResponse("eye_infection_test.html", {"request": request, "username": username})
    else:
        return templates.TemplateResponse("not_authorized.html", {"request": request, "username": username})

@eye_infection_router.post("/predict_WebUI", response_class=HTMLResponse)
async def submit_Response(
    # Enter all the paramaters with their datatypes...
    request: Request,
    requestID: str = "TEST",
    age: int = Form(...),
    breed: int = Form(...),
    sex: int = Form(...),
    redness: int = Form(...),
    swelling: int = Form(...),
    discharge: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./eye_infection/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./eye_infection/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(eye_infection_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = (eye_infection_ml_classifier.predict([[age, breed, sex, redness, swelling, discharge]])[0])
    update_ml_user_data(requestID, age, breed, sex, redness, swelling, discharge, ml_prediction)
    prediction = float(ml_prediction + cnn_prediction)
    if(prediction > 0.85):
        # return {"prediction": prediction, "medicine_data": eye_infection_medicine_data}
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": eye_infection_medicine_data})
    else:
        # return {"prediction": prediction}
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": eye_infection_medicine_data})

def update_ml_user_data(requestID, age, breed, sex, redness, swelling, discharge, prediction):
    eye_infection_ml_data = open("./eye_infection/eye_infection_user_data_ml.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(age) + ", " + str(breed) + ", " + str(sex) + ", " +  \
                str(redness) + ", " + str(swelling) + ", " + str(discharge) + ", " + str(prediction)
    print(new_row)
    eye_infection_ml_data.write('\n' + (new_row))
    eye_infection_ml_data.close()
    return True

def update_cnn_user_data(requestID, image_name, prediction):
    eye_infection_cnn_data = open("./eye_infection/eye_infection_user_data_cnn.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(image_name) + ", " + str(prediction) + ", NA" 
    print(new_row)
    eye_infection_cnn_data.write('\n' + (new_row))
    eye_infection_cnn_data.close()
    return True

@eye_infection_router.post("/submit_feedback")
async def submit_feedback(
    requestID: str = Form(...),
    feedback: int = Form(...)
):
    eye_infection_user_feedback = open("./eye_infection/user_feedback.txt", "a")
    new_row = str(requestID) + ", " + str(feedback) 
    print(new_row)
    eye_infection_user_feedback.write('\n' + (new_row))
    eye_infection_user_feedback.close()
    return True