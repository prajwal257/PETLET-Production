from fastapi import FastAPI, File, UploadFile, Depends, APIRouter, Request, Response, Body, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from data_class import fleas_infection_class
from keras.models import load_model
from authorized_users import authorized_users
from data_class import fleas_infection_class
from random import randint
import tensorflow as tf
import pickle
import uuid
import json
import cv2
import numpy as np

templates = Jinja2Templates(directory="pages")

fleas_infection_router = APIRouter(
    prefix="/fleas_infection",
    tags=["Fleas Infection Prediction"]
)

# Loading Model here.
pickle_in = open("./fleas_infection/fleas_infection_cnn.pkl", "rb")
fleas_infection_cnn_classifier = pickle.load(pickle_in)
pickle_in = open("./fleas_infection/fleas_infection_ml.pkl", "rb")
fleas_infection_ml_classifier = pickle.load(pickle_in)
# Loading the medicine data.
fleas_infection_medicine_data = ""
with open('./fleas_infection/medicine_data.json', 'r') as f: 
    fleas_infection_medicine_data = json.load(f)

@fleas_infection_router.post('/predict')
async def fleas_infection_classifier(
    request: Request,
    requestID: str = "TEST",
    itchingandscratching: int = Form(...),
    hairlossorbaldpatches: int = Form(...),
    redorinflamedskin: int = Form(...),
    fleadirtorfleaeggs: int = Form(...),
    biteorscratchwounds: int = Form(...),
    coatlength: int = Form(...),
    coattype: int = Form(...),
    currentseason: int = Form(...),
    location: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./fleas_infection/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./fleas_infection/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(fleas_infection_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = fleas_infection_ml_classifier.predict([[
        itchingandscratching,
        hairlossorbaldpatches,
        redorinflamedskin,
        fleadirtorfleaeggs,
        biteorscratchwounds,
        coatlength,
        coattype,
        currentseason,
        location
    ]])
    prediction = ml_prediction + cnn_prediction
    if(float(prediction) > 0.85):
        return {"prediction": prediction, "medicine_data": fleas_infection_medicine_data}
        # return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": fleas_infection_medicine_data})
    else:
        return {"prediction": prediction}
        # return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": fleas_infection_medicine_data})


@fleas_infection_router.get("/WebUI/{username}", response_class=HTMLResponse)
async def renderHTML(request: Request, username: str):
    if(username in authorized_users):
        return templates.TemplateResponse("fleas_infection_test.html", {"request": request, "username": username})
    else:
        return templates.TemplateResponse("not_authorized.html", {"request": request, "username": username})
    

@fleas_infection_router.post("/predict_WebUI", response_class=HTMLResponse)
async def submit_Response(
    request: Request,
    requestID: str = "TEST",
    itchingandscratching: int = Form(...),
    hairlossorbaldpatches: int = Form(...),
    redorinflamedskin: int = Form(...),
    fleadirtorfleaeggs: int = Form(...),
    biteorscratchwounds: int = Form(...),
    coatlength: int = Form(...),
    coattype: int = Form(...),
    currentseason: int = Form(...),
    location: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./fleas_infection/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./fleas_infection/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(fleas_infection_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = fleas_infection_ml_classifier.predict([[
        itchingandscratching,
        hairlossorbaldpatches,
        redorinflamedskin,
        fleadirtorfleaeggs,
        biteorscratchwounds,
        coatlength,
        coattype,
        currentseason,
        location
    ]])
    prediction = ml_prediction + cnn_prediction
    if(float(prediction) > 0.85):
        # return {"prediction": prediction, "medicine_data": fleas_infection_medicine_data}
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": fleas_infection_medicine_data})
    else:
        # return {"prediction": prediction}
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": fleas_infection_medicine_data})

def update_ml_user_data(requestID, itchingandscratching, hairlossorbaldpatches, redorinflamedskin, fleadirtorfleaeggs, 
    biteorscratchwounds, coatlength, coattype, currentseason, location, prediction):
    fleas_infection_ml_data = open("./fleas_infection/fleas_infection_user_data_ml.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(itchingandscratching) + ", " + str(hairlossorbaldpatches) + ", " + str(redorinflamedskin) + ", " +  \
                str(fleadirtorfleaeggs) + ", " + str(coatlength) + ", " + str(coattype) + ", " + str(currentseason) + ", " + str(location) + ", " + str(prediction)
    print(new_row)
    fleas_infection_ml_data.write('\n' + (new_row))
    fleas_infection_ml_data.close()
    return True

def update_cnn_user_data(requestID, image_name, prediction):
    fleas_infection_cnn_data = open("./fleas_infection/fleas_infection_user_data_cnn.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(image_name) + ", " + str(prediction) + ", NA" 
    print(new_row)
    fleas_infection_cnn_data.write('\n' + (new_row))
    fleas_infection_cnn_data.close()
    return True

@fleas_infection_router.post("/submit_feedback")
async def submit_feedback(
    requestID: str = Form(...),
    feedback: int = Form(...)
):
    fleas_infection_user_feedback = open("./fleas_infection/user_feedback.txt", "a")
    new_row = str(requestID) + ", " + str(feedback) 
    print(new_row)
    fleas_infection_user_feedback.write('\n' + (new_row))
    fleas_infection_user_feedback.close()
    return True