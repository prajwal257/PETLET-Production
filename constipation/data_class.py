from pydantic import BaseModel
class constipation_class(BaseModel):
    infrequent_or_absent_bowel_movements: int
    small_hard_dry_stools               : int
    visible_discomfort_in_abdomen       : int
    lack_of_appetite                    : int
    lethargy_or_unusual_behavior        : int
    vomiting                            : int