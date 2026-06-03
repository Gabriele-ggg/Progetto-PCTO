from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Modello per le richieste di domande all'API."""
    question: str
