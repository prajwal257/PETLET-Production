from fastapi import FastAPI, File, UploadFile, Depends, APIRouter, Request, Response, Body, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from data_class import constipation_class
from authorized_users import authorized_users
from keras.models import load_model
from data_class import constipation_class
from random import randint
import tensorflow as tf
import pickle
import uuid
import json
import cv2
import numpy as np

templates = Jinja2Templates(directory="pages")

constipation_router = APIRouter(
    prefix="/constipation",
    tags=["Constipation Prediction"]
)

# Loading Model here.
constipation_cnn_classifier = load_model("./constipation/constipation_cnn.h5")
pickle_in = open("./constipation/constipation_ml.pkl", "rb")
constipation_ml_classifier = pickle.load(pickle_in)
# Loading the medicine data.
constipation_medicine_data = ""
with open('./constipation/medicine_data.json', 'r') as f: 
    constipation_medicine_data = json.load(f)

@constipation_router.post('/predict')
async def predict(
    # Enter all the paramaters with their datatypes...
    request: Request,
    requestID: str = "TEST",
    infrequent_or_absent_bowel_movements: int = Form(...),
    small_hard_dry_stools: int = Form(...),
    visible_discomfort_in_abdomen: int = Form(...),
    lack_of_appetite: int = Form(...),
    lethargy_or_unusual_behavior: int = Form(...),
    vomiting: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./constipation/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./constipation/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(constipation_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = (constipation_ml_classifier.predict([[
        infrequent_or_absent_bowel_movements, 
        small_hard_dry_stools, 
        visible_discomfort_in_abdomen, 
        lack_of_appetite, 
        lethargy_or_unusual_behavior, 
        vomiting
    ]])[0])
    update_ml_user_data(
        requestID, infrequent_or_absent_bowel_movements, 
        small_hard_dry_stools, 
        visible_discomfort_in_abdomen, 
        lack_of_appetite, 
        lethargy_or_unusual_behavior, 
        vomiting, 
        ml_prediction
    )
    prediction = float(ml_prediction + cnn_prediction)
    if(prediction > 0.85):
        return {"prediction": prediction, "medicine_data": constipation_medicine_data}
        # return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": constipation_medicine_data})
    else:
        return {"prediction": prediction}
        # return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": constipation_medicine_data})

@constipation_router.post("/predict_WebUI", response_class=HTMLResponse)
async def render_HTML(
    # Enter all the paramaters with their datatypes...
    request: Request,
    requestID: str = "TEST",
    infrequent_or_absent_bowel_movements: int = Form(...),
    small_hard_dry_stools: int = Form(...),
    visible_discomfort_in_abdomen: int = Form(...),
    lack_of_appetite: int = Form(...),
    lethargy_or_unusual_behavior: int = Form(...),
    vomiting: int = Form(...),
    file: UploadFile = File(...)
):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    with open(f"./constipation/images/{file.filename}", "wb") as f:
        f.write(contents)
    image = "./constipation/images/" + file.filename
    image = cv2.imread(image)
    resize = tf.image.resize(image, (256,256))
    cnn_prediction = str(constipation_cnn_classifier.predict(np.expand_dims(resize/255, 0))[0])
    size_of_prediction = len(str(cnn_prediction))
    cnn_prediction = str(cnn_prediction)[1:(size_of_prediction-1)]
    cnn_prediction = float(cnn_prediction)
    update_cnn_user_data(requestID, file.filename, cnn_prediction)
    ml_prediction = (constipation_ml_classifier.predict([[
        infrequent_or_absent_bowel_movements, 
        small_hard_dry_stools, 
        visible_discomfort_in_abdomen, 
        lack_of_appetite, 
        lethargy_or_unusual_behavior, 
        vomiting
    ]])[0])
    update_ml_user_data(
        requestID, 
        infrequent_or_absent_bowel_movements, 
        small_hard_dry_stools, 
        visible_discomfort_in_abdomen, 
        lack_of_appetite, 
        lethargy_or_unusual_behavior, 
        vomiting, 
        ml_prediction
    )
    prediction = float(ml_prediction + cnn_prediction)
    if(prediction > 0.85):
        # return {"prediction": prediction, "medicine_data": constipation_medicine_data}
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": constipation_medicine_data})
    else:
        # return {"prediction": prediction}
        return templates.TemplateResponse("results.html", {"request": request, "prediction": prediction, "medicine_data": constipation_medicine_data})


@constipation_router.get("/WebUI/{username}", response_class=HTMLResponse)
async def renderHTML(request: Request, username: str):
    if(username in authorized_users):
        return templates.TemplateResponse("constipation_test.html", {"request": request, "username": username})
    else:
        return templates.TemplateResponse("not_authorized.html", {"request": request, "username": username})


def update_ml_user_data(
    requestID, 
    infrequent_or_absent_bowel_movements, 
    small_hard_dry_stools, 
    visible_discomfort_in_abdomen, 
    lack_of_appetite, 
    lethargy_or_unusual_behavior, 
    vomiting, 
    prediction
):
    constipation_ml_data = open("./constipation/constipation_user_data_ml.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(infrequent_or_absent_bowel_movements) + ", " + str(small_hard_dry_stools) + ", " + str(visible_discomfort_in_abdomen) + ", " +  \
                str(lack_of_appetite) + ", " + str(lethargy_or_unusual_behavior) + ", " + str(vomiting) + str(prediction)
    print(new_row)
    constipation_ml_data.write('\n' + (new_row))
    constipation_ml_data.close()
    return True

def update_cnn_user_data(requestID, image_name, prediction):
    constipation_cnn_data = open("./constipation/constipation_user_data_cnn.txt", "a")
    new_row = str(requestID) + ", "+ "NA" + ", " + str(image_name) + ", " + str(prediction) + ", NA" 
    print(new_row)
    constipation_cnn_data.write('\n' + (new_row))
    constipation_cnn_data.close()
    return True

@constipation_router.post("/submit_feedback")
async def submit_feedback(
    requestID: str = Form(...),
    feedback: int = Form(...)
):
    constipation_user_feedback = open("./constipation/user_feedback.txt", "a")
    new_row = str(requestID) + ", " + str(feedback) 
    print(new_row)
    constipation_user_feedback.write('\n' + (new_row))
    constipation_user_feedback.close()
    return True