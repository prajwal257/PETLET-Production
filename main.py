# Importing the Dependencies here.
from fastapi import FastAPI, File, UploadFile, Depends, APIRouter, Request, Response, Body, Form
from fastapi.templating import Jinja2Templates
from authorized_users import authorized_users
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from keras.models import load_model
from random import randint
import tensorflow as tf
import numpy as np
import pickle
import uuid
import json
import cv2

# Importing the Routers here.
from tooth_infection.tooth_infection_route import tooth_infection_router
from fleas_infection.fleas_infection_route import fleas_infection_router
from eye_infection.eye_infection_route import eye_infection_router
from ear_infection.ear_infection_route import ear_infection_router
from constipation.constipation_route import constipation_router
from feedback.feedback_route import feedback_router
from jaundice.jaundice_route import jaundice_router
from diarrhea.diarrhea_route import diarrhea_router
from obesity.obesity_route import obesity_router

app = FastAPI()

app.mount("/static", StaticFiles(directory="pages"), name="static")

templates = Jinja2Templates(directory="pages")
@app.get("/home/WebUI", response_class=HTMLResponse)
async def render_HTML(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

templates = Jinja2Templates(directory="pages")
@app.post("/home/WebUI/authenticate_username", response_class=HTMLResponse)
async def render_HTML(request: Request, username: str = Form(...)):
    if(username in authorized_users):
        return templates.TemplateResponse("welcome.html", {"request": request, "username": username})
    else:
        return templates.TemplateResponse("not_authorized.html", {"request": request, "username": username})

app.include_router(feedback_router)
app.include_router(diarrhea_router)
app.include_router(jaundice_router)
app.include_router(constipation_router)
app.include_router(eye_infection_router)
app.include_router(fleas_infection_router)
app.include_router(obesity_router)
app.include_router(tooth_infection_router)
app.include_router(ear_infection_router)