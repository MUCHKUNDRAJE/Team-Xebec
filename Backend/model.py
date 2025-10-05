from pydantic import BaseModel

class AlertRequest(BaseModel):
    email: str
    cme_data: dict