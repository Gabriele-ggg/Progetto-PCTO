"""
api_server.py
=============
Server FastAPI unificato (ex api_server.py + api_server_v2.py).

Endpoint disponibili:
  POST /ask                   → risposta con modello selezionato (accetta QuestionRequest)
  POST /api/chat              → risposta chat (accetta {message|question})
  GET  /api/status            → stato sistema + statistiche linee
  POST /api/set-model         → cambia modello LLM a runtime
  GET  /api/list-models       → lista modelli Ollama disponibili
  POST /api/telemetry         → salva evento di telemetria
  GET  /api/telemetry/summary → sommario per utente
  GET  /api/telemetry/metrics → medie e massimi latenza
  POST /api/upload            → carica PDF in una categoria
  GET  /                      → info API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.models import QuestionRequest
from backend.services.rag_service import (
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
import time
import logging
from datetime import datetime

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
)
logging.getLogger('uvicorn').setLevel(logging.DEBUG)
logging.getLogger('uvicorn.error').setLevel(logging.DEBUG)
logging.getLogger('uvicorn.access').setLevel(logging.DEBUG)

# ─── Percorsi ─────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_CSV = os.path.join(ROOT_DIR, 'data', 'telemetry.csv')

# ─── Inizializzazione globale ─────────────────────────────────────────────────
# Se SKIP_INIT=1 (impostato da demo_runner.py dopo aver già chiamato
# initialize_system() in fase 0), salta la rigenerazione dei dati.
if os.environ.get('SKIP_INIT', '0') == '1':
    print('[INFO] SKIP_INIT=1: salto initialize_system() (dati già generati da demo_runner).')
    DB_INSTANCE = None  # ChromaDB non disponibile in questa modalità; non serve per il RAG JSON-based
else:
    try:
        DB_INSTANCE = initialize_system()
        print('✅ Sistema inizializzato correttamente')
    except Exception as e:
        print(f"❌ ERRORE ALL'AVVIO DEL BACKEND: {repr(e)}")
        DB_INSTANCE = None

# Imposta il modello LLM da variabile d'ambiente (se presente)
_startup_model = os.environ.get('RAG_LLM_MODEL', '').strip()
if _startup_model:
    set_model(_startup_model)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title='EduGuide AI - API', version='2.0')

@app.on_event('shutdown')
def _shutdown_event():
    try:
        unload_model()
    except Exception:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _load_transport_json() -> dict:
    """Carica trasporti.json e ritorna il dict; lancia eccezione se non trovato."""
    path = os.path.join(ROOT_DIR, 'data', 'trasporti.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _categorize_linee(linee: list) -> dict:
    """
    Conta le linee per categoria.
    Prima cerca il campo 'categoria', poi usa classificazione euristica
    (retrocompatibilità con JSON vecchi).
    """
    counts = {'treni': 0, 'urbani': 0, 'extraurbani': 0, 'autobus': 0}
    for linea in linee:
        cat = linea.get('categoria', '').lower()
        if cat in counts:
            counts[cat] += 1
            continue
        desc = (
            linea.get('a', {}).get('p', '') + ' ' +
            linea.get('r', {}).get('p', '')
        ).lower()
        name = str(linea.get('n', '')).lower()
        if 'treno' in desc or 'ferrov' in desc:
            counts['treni'] += 1
        elif 'extra' in desc or 'extraurb' in desc or 'extra' in name:
            counts['extraurbani'] += 1
        elif 'urb' in desc or 'citt' in desc:
            counts['urbani'] += 1
        else:
            counts['autobus'] += 1
    return counts


def _write_telemetry(user: str, session: str, event: str,
                     detail: str, latency_ms, model: str) -> None:
    """
    Scrive una riga nel CSV di telemetria.
    Formato: timestamp|user|session|event|detail|latency_ms|model  (7 colonne)
    """
    os.makedirs(os.path.dirname(TELEMETRY_CSV), exist_ok=True)
    with open(TELEMETRY_CSV, 'a', encoding='utf-8') as f:
        f.write(
            f"{datetime.now().isoformat()}|{user}|{session}|{event}|"
            f"{str(detail).replace('|', ' ')}|{latency_ms}|{model}\n"
        )


def _build_response(response_obj, latency_ms: float) -> tuple[str, int, float]:
    """Normalizza il risultato di generate_travel_response in (testo, tokens, tok/s)."""
    if isinstance(response_obj, dict):
        text   = response_obj.get('response', '')
        tokens = response_obj.get('tokens', 0)
    else:
        text   = str(response_obj)
        tokens = len(text.split())
    tps = round(tokens / (latency_ms / 1000), 1) if latency_ms > 0 and tokens > 0 else 0
    return text, max(0, tokens), max(0, tps)


# ─── Endpoint: /ask ───────────────────────────────────────────────────────────

@app.post('/ask')
def ask_question(question_data: QuestionRequest):
    """
    Endpoint principale (accetta QuestionRequest con campo 'question').
    Usato da api_client.py e da app.py tramite post_ask().
    """
    try:
        t0             = time.time()
        transport_data = get_transport_data()
        context_info   = get_time_aware_context()
        response_obj   = generate_travel_response(
            question=question_data.question,
            transport_data=transport_data,
            context_info=context_info,
        )
        latency_ms               = round((time.time() - t0) * 1000, 2)
        text, tokens, tps        = _build_response(response_obj, latency_ms)

        try:
            _write_telemetry(
                user='api', session='', event='question',
                detail=question_data.question[:50],
                latency_ms=latency_ms,
                model=get_selected_model() or '',
            )
        except Exception as log_err:
            print(f'[WARN] Telemetria non salvata: {log_err}')

        return {
            'response':          text,
            'latency_ms':        latency_ms,
            'tokens':            tokens,
            'tokens_per_second': tps,
            'role':              'EduGuide AI',
            'model':             get_selected_model(),
            'timestamp':         datetime.now().isoformat(),
        }
    except Exception as e:
        print(f'[ERROR] /ask: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ─── Endpoint: /api/chat ──────────────────────────────────────────────────────

@app.post('/api/chat')
async def chat_endpoint(body: dict):
    """
    Endpoint chat principale (accetta {message} o {question}).
    Usato dal frontend HTML/JS.
    """
    message = (body.get('message') or body.get('question') or '').strip()
    if not message:
        raise HTTPException(status_code=400, detail="Messaggio vuoto (invia 'message' o 'question')")

    try:
        t0             = time.time()
        transport_data = get_transport_data()
        context_info   = get_time_aware_context()
        result         = generate_travel_response(
            question=message,
            transport_data=transport_data,
            context_info=context_info,
        )
        latency_ms        = round((time.time() - t0) * 1000, 2)
        text, tokens, tps = _build_response(result, latency_ms)

        try:
            _write_telemetry(
                user='api', session='', event='question',
                detail=message[:50],
                latency_ms=latency_ms,
                model=get_selected_model() or '',
            )
        except Exception as log_err:
            print(f'[WARN] Telemetria non salvata: {log_err}')

        return {
            'response':          text,
            'latency_ms':        latency_ms,
            'tokens':            tokens,
            'tokens_per_second': tps,
            'timestamp':         datetime.now().isoformat(),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'[ERROR] /api/chat: {e}')
        raise HTTPException(status_code=500, detail=f'Errore elaborazione: {str(e)}')


# ─── Endpoint: /api/status ────────────────────────────────────────────────────

@app.get('/api/status')
def get_status():
    """Stato e statistiche per la sidebar/front-end."""
    try:
        data        = _load_transport_json()
        linee       = data.get('linee', [])
        total_corse = sum(
            len(l.get('a', {}).get('f', [])) + len(l.get('r', {}).get('f', []))
            for l in linee
        )
        linee_ids = [str(l.get('n', '')) for l in linee if l.get('n')]
        counts    = _categorize_linee(linee)
        model     = get_selected_model()
        return {
            'status':       'online',
            'db_ready':     DB_INSTANCE is not None,
            'totale_linee': len(linee),
            'totale_corse': total_corse,
            'generato_il':  data.get('orario_dal', ''),
            'linee':        linee_ids,
            'tipi':         counts,
            'backend':      'EduGuide AI',
            'version':      '2.0',
            'model':        str(model) if model else '',
        }
    except FileNotFoundError:
        return {
            'status': 'error', 'db_ready': False,
            'totale_linee': 0, 'totale_corse': 0,
            'tipi': {}, 'linee': [],
            'error': 'trasporti.json non trovato — verificare la cartella data/',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# ─── Endpoint: /api/set-model ─────────────────────────────────────────────────

@app.post('/api/set-model')
def set_model_endpoint(body: dict):
    model_name = (body.get('model') or '').strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Campo 'model' mancante o vuoto")
    try:
        set_model(model_name)
        return {'status': 'ok', 'model': get_selected_model()}
    except Exception as e:
        print(f'[ERROR] /api/set-model: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ─── Endpoint: /api/list-models ───────────────────────────────────────────────

@app.get('/api/list-models')
def list_models():
    import requests as _requests
    models = []
    try:
        resp = _requests.get('http://localhost:11434/api/tags', timeout=5)
        if resp.status_code == 200:
            for m in resp.json().get('models', []):
                name = m.get('name', '')
                if name:
                    models.append(name)
        else:
            print(f'[WARN] Ollama API risponde con {resp.status_code}')
    except Exception as e:
        print(f'[WARN] /api/list-models: Impossibile contattare Ollama: {e}')
    return {'models': models, 'current': get_selected_model()}


# ─── Endpoint: /api/telemetry (POST) ─────────────────────────────────────────

@app.post('/api/telemetry')
def telemetry(event: dict):
    """Riceve eventi di telemetria dal frontend e li scrive su CSV."""
    try:
        _write_telemetry(
            user=str(event.get('user', 'anonimo')),
            session=str(event.get('session', '')),
            event=str(event.get('event', 'unknown')),
            detail=str(event.get('detail', '')),
            latency_ms=event.get('latency_ms', ''),
            model=str(event.get('model', '')),
        )
        return {'status': 'ok'}
    except Exception as e:
        print(f'[ERROR] telemetry: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ─── Endpoint: /api/telemetry/summary ────────────────────────────────────────

@app.get('/api/telemetry/summary')
def telemetry_summary():
    """Sommario per utente: login, domande, sessioni, latenza massima."""
    try:
        if not os.path.exists(TELEMETRY_CSV):
            return {'summary': {}, 'count_lines': 0}

        summary = {}
        with open(TELEMETRY_CSV, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 4:
                    continue
                ts            = parts[0]
                user          = parts[1] if len(parts) > 1 else ''
                session       = parts[2] if len(parts) > 2 else ''
                ev            = parts[3] if len(parts) > 3 else ''
                latency_field = parts[5] if len(parts) > 5 else ''
                model_field   = parts[6] if len(parts) > 6 else ''

                u   = user or 'anonimo'
                rec = summary.setdefault(u, {
                    'logins': 0, 'questions': 0,
                    'sessions': set(), 'last_seen': ts,
                    'models': set(), 'max_latency_ms': None,
                })
                if session:
                    rec['sessions'].add(session)
                if ev.lower().startswith('login'):
                    rec['logins'] += 1
                if ev.lower().startswith('question'):
                    rec['questions'] += 1
                if model_field:
                    rec['models'].add(model_field)
                try:
                    if latency_field:
                        lat = float(latency_field)
                        cur = rec['max_latency_ms']
                        if cur is None or lat > cur:
                            rec['max_latency_ms'] = lat
                except Exception:
                    pass
                if ts > rec.get('last_seen', ''):
                    rec['last_seen'] = ts

        for v in summary.values():
            v['sessions'] = len(v['sessions']) if isinstance(v.get('sessions'), set) else 0
            v['models']   = list(v['models'])   if isinstance(v.get('models'),   set) else []

        count = sum(1 for _ in open(TELEMETRY_CSV, 'r', encoding='utf-8'))
        return {'summary': summary, 'count_lines': count}
    except Exception as e:
        print(f'[ERROR] telemetry_summary: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ─── Endpoint: /api/telemetry/metrics ────────────────────────────────────────

@app.get('/api/telemetry/metrics')
def telemetry_metrics():
    """Metriche aggregate: media e massimo latenza, modelli usati."""
    try:
        if not os.path.exists(TELEMETRY_CSV):
            return {
                'overall': {'avg_latency_ms': None, 'max_latency_ms': None, 'count': 0},
                'by_user': {}, 'models': [],
            }

        overall_latencies = []
        models_set        = set()
        users             = {}

        with open(TELEMETRY_CSV, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 6:
                    continue
                user          = parts[1] or 'anonimo'
                latency_field = parts[5] if len(parts) > 5 else ''
                model_field   = parts[6] if len(parts) > 6 else ''

                try:
                    lat = float(latency_field) if latency_field not in (None, '', 'n/d') else None
                except Exception:
                    lat = None

                if model_field:
                    models_set.add(model_field)

                u = users.setdefault(user, {'latencies': [], 'models': set(), 'count': 0})
                u['count'] += 1
                if lat is not None:
                    overall_latencies.append(lat)
                    u['latencies'].append(lat)
                if model_field:
                    u['models'].add(model_field)

        if overall_latencies:
            overall = {
                'avg_latency_ms': sum(overall_latencies) / len(overall_latencies),
                'max_latency_ms': max(overall_latencies),
                'count':          len(overall_latencies),
            }
        else:
            overall = {'avg_latency_ms': None, 'max_latency_ms': None, 'count': 0}

        by_user = {}
        for user, info in users.items():
            lat_list = info['latencies']
            by_user[user] = {
                'avg_latency_ms': (sum(lat_list) / len(lat_list)) if lat_list else None,
                'max_latency_ms': max(lat_list) if lat_list else None,
                'count':          info['count'],
                'models':         list(info['models']),
            }

        return {'overall': overall, 'by_user': by_user, 'models': sorted(models_set)}
    except Exception as e:
        print(f'[ERROR] telemetry_metrics: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ─── Endpoint: /api/upload ────────────────────────────────────────────────────

VALID_CATEGORIES = {'urbani', 'extraurbani', 'treni', 'circolari_scuola'}

@app.post('/api/upload')
async def upload_files(
    files: list[UploadFile] = File(...),
    category: str = Form(...),
):
    """Carica file PDF in una delle 4 categorie: urbani, extraurbani, treni, circolari_scuola."""
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Categoria non valida. Usa una tra: {', '.join(sorted(VALID_CATEGORIES))}",
        )

    upload_dir = os.path.join(ROOT_DIR, 'data', 'pdfs', category)
    os.makedirs(upload_dir, exist_ok=True)

    saved  = 0
    errors = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith('.pdf'):
            errors.append(f"Saltato '{f.filename}': non è un PDF")
            continue
        safe_name = (
            ''.join(c for c in f.filename if c.isalnum() or c in '._- ')
            .strip().replace(' ', '_') or 'file.pdf'
        )
        dest = os.path.join(upload_dir, safe_name)
        try:
            content = await f.read()
            with open(dest, 'wb') as out:
                out.write(content)
            saved += 1
            print(f'[OK] Upload {category}/{safe_name} ({len(content)} bytes)')
        except Exception as e:
            errors.append(f"Errore su '{f.filename}': {e}")

    if saved == 0:
        raise HTTPException(
            status_code=500,
            detail='Nessun file salvato. Errori: ' + '; '.join(errors),
        )

    return {'status': 'ok', 'category': category, 'saved': saved, 'errors': errors}


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get('/')
def root():
    return {
        'name':      'EduGuide AI API',
        'version':   '2.0',
        'endpoints': {
            'status':    '/api/status',
            'chat':      '/api/chat',
            'ask':       '/ask',
            'set_model': '/api/set-model',
            'upload':    '/api/upload',
        },
    }