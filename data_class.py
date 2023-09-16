from pydantic import BaseModel
class diarrhea_class(BaseModel):
    requestID: str
    age : int
    blood_presence : int
    consistency : int
    diet_changes : int
    breed : int

class jaundice_class(BaseModel):
    requestID              : str
    vomiting               : int
    diarrhoea              : int
    lethargy               : int
    fever                  : float
    abdominal_pain         : int
    loss_of_appetite       : int
    paleness               : int
    yellowish_skin         : int
    change_in_urine_feces  : int
    polyuria               : int
    polydipsia             : int
    mental_confusion       : int
    weight_loss            : float
    bleeding               : int

class Options (BaseModel):
    FileName: str
    FileDesc: str = "Upload for demonstration"

class eye_infection_class(BaseModel):
    Age : int
    Breed : int
    Sex : int
    Redness : int
    Swelling : int
    Discharge : int

class fleas_infection_class(BaseModel):
    itchingandscratching : int
    hairlossorbaldpatches : int
    redorinflamedskin : int
    fleadirtorfleaeggs : int
    biteorscratchwounds : int
    coatlength : int
    coattype : int
    currentseason : int
    location : int

class constipation_class(BaseModel):
    infrequent_or_absent_bowel_movements: int
    small_hard_dry_stools               : int
    visible_discomfort_in_abdomen       : int
    lack_of_appetite                    : int
    lethargy_or_unusual_behavior        : int
    vomiting                            : int

class feedback_class(BaseModel):
    requestID                           : int
    feedback_score                      : int