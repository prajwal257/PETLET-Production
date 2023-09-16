from pydantic import BaseModel
class jaundice_class(BaseModel):
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