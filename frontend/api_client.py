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
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Impossibile connettersi al Backend API. Assicurati che FastAPI sia avviato sulla porta 8000."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Errore HTTP dal server: {e.response.json().get('detail', 'Errore sconosciuto')}"}


def get_telemetry_summary() -> dict:
    """Recupera il sommario di telemetria dal backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/telemetry/summary", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Impossibile recuperare la telemetria: {e}"}


def get_telemetry_metrics() -> dict:
    """Recupera le metriche di telemetria estese (min/media/max per modello e per utente)."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/telemetry/metrics", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Impossibile recuperare le metriche: {e}"}


def upload_files(files: list, category: str) -> dict:
    """Invia una lista di file al backend nella categoria specificata."""
    try:
        multipart = []
        for f in files:
            content = f.read()
            multipart.append(("files", (f.name, content, "application/pdf")))
        data = {"category": category}
        resp = requests.post(f"{BACKEND_URL}/api/upload", files=multipart, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        try:
            return {"error": e.response.json()}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def post_feedback(user: str, session: str, role: str, text: str, feedback: str) -> dict:
    """Invia un feedback (positive/negative/removed) per un messaggio specifico."""
    payload = {
        "user": user,
        "session": session,
        "role": role,
        "text": text[:200],
        "feedback": feedback,
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/api/feedback", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}