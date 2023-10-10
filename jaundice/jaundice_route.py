from data_class import jaundice_class, feedback_class
import pandas as pd
import pickle
import json
from fastapi import APIRouter

jaundice_router = APIRouter(
    prefix="/jaundice",
    tags=["Jaundice Prediction"]
)
# Loading jaundice Model here.
pickle_in = open("./jaundice/jaundice.pkl", "rb")
jaundice_classifier = pickle.load(pickle_in)
jaundice_medicine_data = ""
with open('./jaundice/medicine_data.json', 'r') as f: 
    jaundice_medicine_data = json.load(f)
@jaundice_router.post("/predict")
def predict(data : jaundice_class):
    requestID                   = str(data.requestID)
    vomiting                    = int(data.vomiting)  
    diarrhoea                   = int(data.diarrhoea)  
    lethargy                    = int(data.lethargy)  
    fever                       = float(data.fever)
    abdominal_pain              = int(data.abdominal_pain)  
    loss_of_appetite            = int(data.loss_of_appetite)  
    paleness                    = int(data.paleness)  
    yellowish_skin              = int(data.yellowish_skin)  
    change_in_urine_feces       = int(data.change_in_urine_feces)  
    polyuria                    = int(data.polyuria)  
    polydipsia                  = int(data.polydipsia)  
    mental_confusion            = int(data.mental_confusion)  
    weight_loss                 = float(data.weight_loss)
    bleeding                    = int(data.bleeding)  
    prediction                  = int(jaundice_classifier.predict([[vomiting, diarrhoea, lethargy, fever, abdominal_pain, loss_of_appetite, paleness, yellowish_skin, change_in_urine_feces, polyuria, polydipsia, mental_confusion, weight_loss, bleeding]]))
    jaundice_data               = open("jaundice/jaundice_user_data.txt", "a")
    new_row                     = str(vomiting) + ", " + str(diarrhoea) + ", " + str(lethargy) + ", " + str(fever) + ", " + \
                                    str(abdominal_pain) + ", " +  str(loss_of_appetite) + ", " + str(paleness) + ", " + \
                                    str(yellowish_skin) + ", " +  str(change_in_urine_feces) + ", " + str(polyuria) + ", " + \
                                    str(polydipsia) + ", " + str(mental_confusion) + ", " + str(weight_loss) + ", " + str(bleeding) + ", " + \
                                    str(prediction) + ", NA \n" 
    print(new_row)
    jaundice_data.write((new_row))
    jaundice_data.close()
    if(prediction > 0.5):
        # Return the prediction with medicines.
        return {"prediction": prediction, "medicine_data": jaundice_medicine_data}
    else:
        return {"prediction": prediction}

@jaundice_router.post("/feedback")
async def submit_feedback(data: feedback_class):
    jaundice_feedback   = open("./jaundice/feedback_user_data.txt", "a")
    requestID = str(data.requestID)
    feedback_score = str(data.feedback_score)
    new_row = requestID + ", " + feedback_score + "\n"
    print(new_row)
    jaundice_feedback.write(new_row)
    return True