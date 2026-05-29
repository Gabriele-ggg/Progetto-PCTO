from pydantic import BaseModel, EmailStr
from datetime import datetime

class Convert(BaseModel):
    id: int
    username: str
    signup_ts: datetime | None = None
    email: EmailStr

external_data = {
    "id": "123",  
    "username": "mario_rossi",
    "signup_ts": "2026-05-29 12:00",  
    "email": "mario@example.com"
}

converted = Convert.model_validate(external_data)

# Verifichiamo il tipo del dato convertito prendendolo direttamente dal processo di validazione
print(type(converted.id))
# Output: <class 'int'>