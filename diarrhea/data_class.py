from pydantic import BaseModel
class diarrhea_class(BaseModel):
    requestID: int
    age : int
    blood_presence : int
    consistency : int
    diet_changes : int
    breed : int