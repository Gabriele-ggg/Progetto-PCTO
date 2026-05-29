"""
backend_test.py — Giorno 1
API base con FastAPI per testare endpoint REST e interazione con Ollama (LLM locale).

Avvio:
    uvicorn backend_test:app --reload

Dipendenze:
    pip install fastapi uvicorn requests
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="AI Backend Test", version="1.0")

# ---------------------------------------------------------------------------
# Modelli Pydantic (struttura JSON in input/output)
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:1.5b"   # modello Ollama di default
    temperature: float = 0.7


class PromptResponse(BaseModel):
    model: str
    response: str
    prompt_tokens: int | None = None


# ---------------------------------------------------------------------------
# Endpoint di health-check
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Verifica che il server sia attivo."""
    return {"status": "ok", "message": "Backend AI attivo"}


@app.get("/health")
def health():
    """Controlla se Ollama è raggiungibile."""
    try:
        res = requests.get("http://localhost:11434", timeout=3)
        ollama_ok = res.status_code == 200
    except requests.ConnectionError:
        ollama_ok = False

    return {
        "fastapi": "running",
        "ollama": "reachable" if ollama_ok else "unreachable"
    }


# ---------------------------------------------------------------------------
# Endpoint principale: invio prompt a Ollama
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=PromptResponse)
def generate(request: PromptRequest):
    """
    Invia un prompt al modello LLM locale tramite Ollama API.
    Ollama espone un'API REST su http://localhost:11434.
    """
    ollama_url = "http://localhost:11434/api/generate"

    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,                      # risposta singola, non streaming
        "options": {
            "temperature": request.temperature
        }
    }

    try:
        res = requests.post(ollama_url, json=payload, timeout=60)
        res.raise_for_status()
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama non raggiungibile. Avvialo con: ollama serve")
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Errore Ollama: {e}")

    data: dict = res.json()

    return PromptResponse(
        model=data.get("model", request.model),
        response=data.get("response", ""),
        prompt_tokens=data.get("prompt_eval_count")
    )


# ---------------------------------------------------------------------------
# Endpoint extra: manipolazione dizionari JSON (esercizio casting)
# ---------------------------------------------------------------------------

class DataInput(BaseModel):
    valori: list  # lista mista di tipi


@app.post("/cast")
def cast_input(data: DataInput):
    """
    Esercizio Giorno 1: riceve una lista di valori e li converte
    in int/float/str a seconda del contenuto.
    Es: ["3", "3.14", "hello", 42] → [3, 3.14, "hello", 42]
    """
    risultati = []
    for v in data.valori:
        cast = _smart_cast(v)
        risultati.append({"originale": v, "tipo": type(cast).__name__, "valore": cast})

    return {"risultati": risultati}


def _smart_cast(valore):
    """Tenta il cast in int, poi float, poi lascia come stringa."""
    if isinstance(valore, (int, float, bool)):
        return valore
    s = str(valore).strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s