from data_class import diarrhea_class, feedback_class
import pandas as pd
import pickle
import json
from fastapi import APIRouter

diarrhea_router = APIRouter(
    prefix="/diarrhea",
    tags=["Diarrhea Prediction"]
)
# Loading diarrhea Model here.
pickle_in = open("./diarrhea/diarrhea.pkl", "rb")
diarrhea_classifier = pickle.load(pickle_in)
diarrheoa_medicine_data = ""
with open('./diarrhea/medicine_data.json', 'r') as f: 
    diarrheoa_medicine_data = json.load(f)

@diarrhea_router.post("/predict")
def predict(data : diarrhea_class):
    requestID       = str(data.requestID)
    age             = int(data.age)
    blood_presence  = int(data.blood_presence)
    consistency     = int(data.consistency)
    diet_changes    = int(data.diet_changes)
    breed           = int(data.breed)
    prediction      = int(diarrhea_classifier.predict([[age, blood_presence, consistency, diet_changes, breed]]))
    diarrhea_data   = open("./diarrhea/diarrhea_user_data.txt", "a")
    new_row         = str(requestID) + ", " + str(age) + ", " + str(blood_presence) + ", " + str(data.consistency) + ", " +  \
                        str(data.diet_changes) + ", " + str(data.breed) + ", " + str(prediction) + ", NA \n" 
    print(new_row)
    diarrhea_data.write(new_row)
    diarrhea_data.close()
    if(prediction > 0.5):
        # Return the prediction with medicines.
        return {"prediction": prediction, "medicine_data": diarrheoa_medicine_data}
    else:
        return {"prediction": prediction}

@diarrhea_router.post("/feedback")
async def submit_feedback(data: feedback_class):
    return true