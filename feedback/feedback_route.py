from fastapi import APIRouter
from data_class import feedback_class

feedback_router = APIRouter(
    prefix="/submit_feedback",
    tags=["Submit Feedback for specific Model"]
)

@feedback_router.post("/submit_feedback")
async def submit_feedback(data: feedback_class):
    feedback_report = open("./feedback/feedback_repot.txt", "a")
    new_row         = str(requestID) + ", " + str(disease) + ", " + str(feedback) + ", NA \n" 
    print(new_row)
    diarrhea_data.write(new_row)
    diarrhea_data.close()