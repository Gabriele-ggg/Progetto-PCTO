import requests
from pydantic import BaseModel

BACKEND_URL = "http://localhost:8000"

class QuestionRequest(BaseModel):
    question: str

def post_ask(question: str) -> dict:
    """Invia la domanda al backend FastAPI e restituisce la risposta."""
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(f"{BACKEND_URL}/ask", json=payload, headers=headers)
        response.raise_for_status() # Solleva eccezione per codici 4xx/5xx
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Impossibile connettersi al Backend API. Assicurati che FastAPI sia avviato sulla porta 8000."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Errore HTTP dal server: {e.response.json().get('detail', 'Errore sconosciuto')}"}

