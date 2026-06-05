from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import os
from datetime import datetime

# Importazioni dal progetto
from .services.rag_service import initialize_system, get_transport_data, generate_travel_response

# --- INIZIALIZZAZIONE ---
ROOT_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    DB_INSTANCE = initialize_system()
    print("✅ Sistema inizializzato correttamente")
except Exception as e:
    print(f"❌ ERRORE ALL'AVVIO: {e}")
    DB_INSTANCE = None

app = FastAPI(title="TPL FVG AI - API", version="2.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _load_json() -> dict:
    """Carica trasporti.json e ritorna il dict; lancia eccezione se non trovato."""
    json_path = os.path.join(ROOT_PROJECT_DIR, "data", "trasporti.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _categorize_linee(linee: list) -> dict:
    """
    Conta le linee per categoria.
    Prima cerca il campo 'categoria', poi usa una classificazione euristica
    basata sulla descrizione del percorso (retrocompatibilità con JSON vecchi).
    """
    counts = {"treni": 0, "urbani": 0, "extraurbani": 0, "autobus": 0}
    for linea in linee:
        cat = linea.get("categoria", "").lower()
        if cat in counts:
            counts[cat] += 1
            continue
        # Fallback euristico
        desc = (
            linea.get("a", {}).get("p", "") + " " +
            linea.get("r", {}).get("p", "")
        ).lower()
        name = str(linea.get("n", "")).lower()
        if "treno" in desc or "ferrov" in desc:
            counts["treni"] += 1
        elif "extra" in desc or "extraurb" in desc or "extra" in name:
            counts["extraurbani"] += 1
        elif "urb" in desc or "citt" in desc:
            counts["urbani"] += 1
        else:
            counts["autobus"] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Stato del database e statistiche per la sidebar."""
    try:
        data  = _load_json()
        linee = data.get("linee", [])

        total_corse = sum(
            len(l.get("a", {}).get("f", [])) + len(l.get("r", {}).get("f", []))
            for l in linee
        )
        counts = _categorize_linee(linee)

        # BUG FIX: campo 'linee' mancante — necessario per la sidebar del frontend
        linee_ids = [str(l.get("n", "")) for l in linee if l.get("n")]

        return {
            "status":       "online",
            "db_ready":     DB_INSTANCE is not None,
            "totale_linee": len(linee),
            "totale_corse": total_corse,
            "generato_il":  data.get("orario_dal", ""),
            "tipi":         counts,
            "linee":        linee_ids,   # ← era mancante: causa "Nessuna linea" nella sidebar
        }

    except FileNotFoundError:
        return {
            "status": "error", "db_ready": False,
            "totale_linee": 0, "totale_corse": 0,
            "tipi": {}, "linee": [],
            "error": "trasporti.json non trovato — verificare la cartella data/",
        }
    except Exception as e:
        return {
            "status": "error", "db_ready": False,
            "totale_linee": 0, "totale_corse": 0,
            "tipi": {}, "linee": [],
            "error": str(e),
        }


@app.post("/api/chat")
async def chat_endpoint(body: dict):
    """Endpoint chat principale."""
    if DB_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Database non inizializzato")

    message = (
        body.get("message", "") or
        body.get("question", "") or
        ""
    ).strip()

    if not message:
        raise HTTPException(status_code=400, detail="Messaggio vuoto (invia 'message' o 'question')")

    try:
        import time
        t0 = time.time()
        transport_data = get_transport_data()
        response_text = generate_travel_response(
            question=message,
            transport_data=transport_data,
            context_info=f"Richiesta: {message[:80]}",
        )
        latency_ms = round((time.time() - t0) * 1000, 2)

        # Log telemetria in modo sicuro
        try:
            log_dir = os.path.join(ROOT_PROJECT_DIR, "data", "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "telemetry.csv"), "a", encoding="utf-8") as fh:
                fh.write(
                    f"{datetime.now().isoformat()}|{latency_ms}|"
                    f"{message[:50].replace('|', ' ')}|"
                    f"{response_text[:100].replace('|', ' ')}\n"
                )
        except Exception as log_err:
            print(f"[WARN] Telemetria non salvata: {log_err}")

        return {
            "response": response_text,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] /api/chat: {e}")
        raise HTTPException(status_code=500, detail=f"Errore elaborazione: {str(e)}")

@app.post("/ask")
def ask_question_legacy(body: dict):
    """Endpoint legacy — delega a /api/chat internamente."""
    import asyncio
    import inspect
    # Chiama la funzione async in modo sincrono
    coro = chat_endpoint(body)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, chat_endpoint(body))
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except Exception:
        # Fallback diretto
        question = (body.get("question") or body.get("message") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Domanda vuota")
        transport_data = get_transport_data()
        response_text  = generate_travel_response(
            question=question, transport_data=transport_data,
            context_info="Legacy endpoint"
        )
        return {"response": response_text, "timestamp": datetime.now().isoformat()}


@app.get("/")
def root():
    return {
        "name": "TPL FVG AI API", "version": "2.0",
        "endpoints": {"status": "/api/status", "chat": "/api/chat", "ask": "/ask"},
    }