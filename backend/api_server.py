from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QuestionRequest
from services.rag_service import (
    initialize_system,
    get_time_aware_context,
    get_transport_data,
    generate_travel_response,
    generate_generic_response,
    get_selected_model,
    set_model,
    unload_model,
)

import os
import json
from datetime import datetime

# --- INIZIALIZZAZIONE GLOBALE DEL SERVIZIO ---
try:
    DB_INSTANCE = initialize_system()
except Exception as e:
    print("ERRORE ALL'AVVIO DEL BACKEND:", repr(e))
    DB_INSTANCE = None

# Imposta il modello LLM da variabile d'ambiente (se presente)
# Esempi:
#   RAG_LLM_MODEL=mistral:7b                           → Ollama locale
_startup_model = os.environ.get("RAG_LLM_MODEL", "").strip()
if _startup_model:
    set_model(_startup_model)

app = FastAPI(title="EduGuide AI - API", version="2.0")

# Ensure model resources are freed on FastAPI shutdown
@app.on_event('shutdown')
def _shutdown_event():
    try:
        unload_model()
    except Exception:
        pass

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility: carica trasporti.json per /api/status
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_transport_json() -> dict:
    path = os.path.join(ROOT_DIR, 'data', 'trasporti.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@app.post('/ask')
def ask_question(question_data: QuestionRequest):
    """Endpoint principale: risponde a qualsiasi domanda usando il modello selezionato."""
    if DB_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Sistema non inizializzato")
    try:
        import time
        t0 = time.time()
        transport_data = get_transport_data()
        context_info   = get_time_aware_context(DB_INSTANCE)
        response_text  = generate_travel_response(
            question=question_data.question,
            transport_data=transport_data,
            context_info=context_info,
        )
        latency_ms = (time.time() - t0) * 1000
        return {
            'response':   response_text,
            'latency_ms': round(latency_ms, 2),
            'role':       'EduGuide AI',
            'model':      get_selected_model(),
            'timestamp':  datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"[ERROR] /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/set-model')
def set_model_endpoint(body: dict):
    """
    Cambia il modello LLM Ollama a runtime senza riavviare il server.

    Body JSON: {"model": "<nome_modello>"}

    Esempi di valori validi per "model":
      "rule-based"   → nessun LLM, solo logica a regole
      "mistral:7b"   → modello Ollama locale
      "llama3:8b"    → modello Ollama locale
    """
    model_name = (body.get("model") or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Campo 'model' mancante o vuoto")

    try:
        set_model(model_name)
        return {"status": "ok", "model": get_selected_model()}
    except Exception as e:
        print(f"[ERROR] /api/set-model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/list-models')
def list_models():
    """
    Restituisce la lista dei modelli disponibili.
    Tenta di contattare l'API di Ollama; se non disponibile, restituisce solo 'rule-based'.
    """
    import requests as _requests
    models = ['rule-based']
    try:
        resp = _requests.get('http://localhost:11434/api/tags', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for model in data.get('models', []):
                name = model.get('name', '')
                if name:
                    models.append(name)
        else:
            print(f"[WARN] Ollama API risponde con {resp.status_code}")
    except Exception as e:
        print(f"[WARN] /api/list-models: Impossibile contattare Ollama (localhost:11434): {e}")
    return {'models': models, 'current': get_selected_model()}


@app.get('/api/status')
def get_status():
    """Stato e statistiche per la sidebar/front-end."""
    try:
        data = _load_transport_json()
        linee = data.get('linee', [])
        total_corse = sum(len(l.get('a', {}).get('f', [])) + len(l.get('r', {}).get('f', [])) for l in linee)
        linee_ids = [str(l.get('n', '')) for l in linee if l.get('n')]
        model = get_selected_model()
        return {
            'status': 'online' if DB_INSTANCE is not None else 'offline',
            'db_ready': DB_INSTANCE is not None,
            'totale_linee': len(linee),
            'totale_corse': total_corse,
            'generato_il': data.get('orario_dal', ''),
            'linee': linee_ids,
            'backend': 'EduGuide AI',
            'version': '2.0',
            'model': str(model) if model else 'rule-based',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.post('/api/chat')
async def chat_endpoint(body: dict):
    """Endpoint chat principale (compatibilità frontend)."""
    if DB_INSTANCE is None:
        raise HTTPException(status_code=503, detail='Database non inizializzato')
    message = (body.get('message') or body.get('question') or '').strip()
    if not message:
        raise HTTPException(status_code=400, detail='Messaggio vuoto')
    try:
        import time
        t0 = time.time()
        transport_data = get_transport_data()
        # per richieste di viaggio usiamo il generatore specifico
        result = generate_travel_response(question=message, transport_data=transport_data, context_info=f"Richiesta: {message[:80]}")
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        # Estrai risposta e token dal risultato
        if isinstance(result, dict):
            response_text = result.get('response', '')
            tokens = result.get('tokens', 0)
        else:
            response_text = result
            tokens = len(response_text.split())
        
        # Calcola token al secondo (con sanity check)
        tokens_per_second = 0
        if latency_ms > 0 and tokens > 0:
            tokens_per_second = round(tokens / (latency_ms / 1000), 1)
        
        return {
            'response': response_text,
            'latency_ms': latency_ms,
            'tokens': max(0, tokens),
            'tokens_per_second': max(0, tokens_per_second),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] /api/chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/telemetry')
def telemetry(event: dict):
    """Riceve eventi di telemetria dal frontend e li scrive su CSV (pipe-separated).
    Body atteso: {user, session, event, detail, latency_ms, model}
    """
    try:
        user = str(event.get('user', 'anonimo'))
        session = str(event.get('session', ''))
        ev = str(event.get('event', 'unknown'))
        detail = str(event.get('detail', ''))
        latency_ms = event.get('latency_ms', '')
        model = event.get('model', '')
        data_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_path, exist_ok=True)
        fp = os.path.join(data_path, 'telemetry.csv')
        with open(fp, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}|{user}|{session}|{ev}|{detail}|{latency_ms}|{model}\n")
        return {'status': 'ok'}
    except Exception as e:
        print(f"[ERROR] telemetry endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/telemetry/summary')
def telemetry_summary():
    try:
        fp = os.path.join(os.getcwd(), 'data', 'telemetry.csv')
        if not os.path.exists(fp):
            return {'summary': {}, 'count_lines': 0}
        summary = {}
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 4:
                    continue
                ts = parts[0]
                user = parts[1] if len(parts) > 1 else ''
                session = parts[2] if len(parts) > 2 else ''
                ev = parts[3] if len(parts) > 3 else ''
                detail = parts[4] if len(parts) > 4 else ''
                latency_field = parts[5] if len(parts) > 5 else ''
                model_field = parts[6] if len(parts) > 6 else ''
                u = user or 'anonimo'
                rec = summary.setdefault(u, {'logins': 0, 'questions': 0, 'sessions': set(), 'last_seen': ts, 'models': set(), 'max_latency_ms': None})
                if session:
                    try:
                        rec['sessions'].add(session)
                    except Exception:
                        pass
                if ev.lower().startswith('login'):
                    rec['logins'] += 1
                if ev.lower().startswith('question'):
                    rec['questions'] += 1
                try:
                    if model_field:
                        rec['models'].add(model_field)
                except Exception:
                    pass
                try:
                    if latency_field:
                        lat = float(latency_field)
                        cur = rec.get('max_latency_ms')
                        if cur is None or lat > cur:
                            rec['max_latency_ms'] = lat
                except Exception:
                    pass
                if ts > rec.get('last_seen', ''):
                    rec['last_seen'] = ts
        for k, v in summary.items():
            if isinstance(v.get('sessions'), set):
                v['sessions'] = len(v['sessions'])
            else:
                v['sessions'] = 0
            if isinstance(v.get('models'), set):
                v['models'] = list(v['models'])
            else:
                v['models'] = []
            if v.get('max_latency_ms') is None:
                v['max_latency_ms'] = None
        return {'summary': summary, 'count_lines': sum(1 for _ in open(fp, 'r', encoding='utf-8'))}
    except Exception as e:
        print(f"[ERROR] telemetry_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
def root():
    return {
        'name': 'EduGuide AI API',
        'version': '2.0',
        'endpoints': {
            'status':    '/api/status',
            'chat':      '/api/chat',
            'ask':       '/ask',
            'set_model': '/api/set-model',
        },
    }