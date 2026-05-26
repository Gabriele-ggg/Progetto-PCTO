#!/usr/bin/env python3
"""
Mini AI TPL FVG — Backend FastAPI + Uvicorn
PDF letti da sottocartelle ./pdf/urbani/, ./pdf/extraurbani/, ./pdf/treni/
Ollama llama3:8b come LLM locale.
Calcola la cadenza mediana per ogni fermata/linea/direzione.
"""

import os, sys, json, re, logging, httpx, statistics
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pypdf import PdfReader

# ═══════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tpl_fvg_debug.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("TPL_FVG")

# ═══════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════
SCRIPT_DIR   = Path(__file__).parent.resolve()
PDF_DIR      = SCRIPT_DIR / "pdf"
JSON_PATH    = SCRIPT_DIR / "tpl_fvg_orari.json"
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:8b"

TRENO_KW  = ["treno","train","intercity","freccia","italo","regionale","trenitalia","fs ","ferroviaria","rail"]
EXTRA_KW  = ["extraurbano","extra-urbano","interurbano","pullman","corriera","extraurb","suburbano"]
URBANO_KW = ["urbano","urban","città","city","bus u","linea u","apu","atap","apt trieste"]

TIME_RE  = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
LINEA_RE = re.compile(
    r"(?:linea|line|bus|autobus|treno|corsa|servizio|n[°\.]?)\s*[:\-]?\s*([A-Za-z0-9/\-]{1,6})",
    re.IGNORECASE,
)
DIR_RE = re.compile(
    r"(?:direzione|dir\.?|verso|capolinea|destinazione|per|da\s+\S+\s+a)\s*[:\-]?\s*([^\n\r,;]{3,80})",
    re.IGNORECASE,
)
NOTE_RE = re.compile(
    r"(?:note|avviso|info|attenzione|orario\s+estivo|orario\s+invernale|festivo|feriale)[:\s]+([^\n\r]{5,120})",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════
# PARSING PDF
# ═══════════════════════════════════════════════

def detect_type(text: str) -> str:
    """Funzione di fallback per determinare il tipo dal testo (usata se non specificato)"""
    tl = text.lower()
    for kw in TRENO_KW:
        if kw in tl: return "treno"
    for kw in EXTRA_KW:
        if kw in tl: return "extraurbano"
    for kw in URBANO_KW:
        if kw in tl: return "urbano"
    return "autobus"


def extract_stops_times(page_text: str) -> list[dict]:
    """
    Estrae TUTTE le coppie fermata/orario dalla pagina.
    Gestisce sia righe con un orario che righe con orari multipli (tabelle).
    """
    stops = []
    seen  = set()
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        times = TIME_RE.findall(line)
        if not times:
            continue
        # Rimuovi gli orari per ottenere il nome fermata
        name = TIME_RE.sub("", line).strip(" |•-–—\t:/")
        name = re.sub(r"\s{2,}", " ", name).strip()
        # Salta righe che sono solo numeri/simboli dopo aver rimosso orari
        if len(name) < 2 or re.fullmatch(r"[\d\s\.\-\|/]+", name):
            continue
        for t in times:
            key = (name, t)
            if key not in seen:
                seen.add(key)
                stops.append({"fermata": name, "orario": t})
    return stops


def extract_raw_text_blocks(page_text: str) -> list[str]:
    """Estrae blocchi di testo significativi (almeno 4 parole) per contesto aggiuntivo."""
    blocks = []
    for line in page_text.splitlines():
        clean = line.strip()
        if len(clean.split()) >= 4 and not TIME_RE.search(clean):
            blocks.append(clean)
    return blocks[:20]  # al massimo 20 righe di contesto per pagina


def parse_pdf(pdf_path: Path, tipo_forzato: str | None = None) -> list[dict]:
    log.debug(f"[PDF] ── Apertura: {pdf_path.name}")
    risultati = []
    try:
        reader  = PdfReader(str(pdf_path))
        n_pages = len(reader.pages)
        log.debug(f"[PDF] {pdf_path.name} → {n_pages} pagine")

        # ── Analisi globale (prime 5 pagine per tipo e metadati) ──
        full_text = "\n".join(
            (reader.pages[i].extract_text() or "") for i in range(min(5, n_pages))
        )
        
        # Determina il tipo: forzato dalla sottocartella o rilevato dal testo
        if tipo_forzato is not None:
            tipo_mezzo = tipo_forzato
            log.debug(f"[PDF] Tipo mezzo forzato (da sottocartella): {tipo_mezzo}")
        else:
            tipo_mezzo = detect_type(full_text)
            log.debug(f"[PDF] Tipo mezzo rilevato dal testo: {tipo_mezzo}")

        # Linea e direzione dall'header globale
        m = LINEA_RE.search(full_text)
        linea_globale = m.group(1).upper().strip() if m else None
        m = DIR_RE.search(full_text)
        dir_globale = m.group(1).strip() if m else None
        log.debug(f"[PDF] Linea globale={linea_globale} Dir globale={dir_globale}")

        # Note globali
        note_globali = []
        for mn in NOTE_RE.finditer(full_text):
            note_globali.append(mn.group(1).strip())

        # ── Analisi pagina per pagina ──
        for i, page in enumerate(reader.pages):
            testo = page.extract_text() or ""
            if not testo.strip():
                log.debug(f"[PDF] Pagina {i+1}: vuota/scansione, skip")
                continue

            # Linea locale
            m = LINEA_RE.search(testo)
            linea = m.group(1).upper().strip() if m else linea_globale

            # Direzione locale
            m = DIR_RE.search(testo)
            direzione = m.group(1).strip() if m else dir_globale

            # Fermate + orari (TUTTE, nessun limite)
            fermate = extract_stops_times(testo)

            # Note locali
            note_pagina = [mn.group(1).strip() for mn in NOTE_RE.finditer(testo)]

            # Contesto testuale aggiuntivo
            contesto = extract_raw_text_blocks(testo)

            # Salta se non c'è nulla di utile
            if not fermate and not linea:
                log.debug(f"[PDF] Pagina {i+1}: nessuna fermata né linea, skip")
                continue

            record = {
                "tipo":      tipo_mezzo,
                "linea":     linea or f"{pdf_path.stem}_P{i+1}",
                "direzione": direzione or "N/D",
                "fermate":   fermate,           # lista completa senza tagli
                "note":      list(dict.fromkeys(note_globali + note_pagina)),  # dedup
                "contesto":  contesto,
                "sorgente":  pdf_path.name,
                "pagina":    i + 1,
            }
            risultati.append(record)
            log.debug(
                f"[PDF] Pagina {i+1}: linea={record['linea']} "
                f"dir={record['direzione']} fermate={len(fermate)} note={len(record['note'])}"
            )

        log.info(f"[PDF] ✅ {pdf_path.name}: {len(risultati)} record estratti")
    except Exception as e:
        log.error(f"[PDF] ❌ Errore {pdf_path.name}: {e}", exc_info=True)
    return risultati


# ═══════════════════════════════════════════════
# GESTIONE JSON
# ═══════════════════════════════════════════════
REQUIRED = {"tipo", "linea", "direzione", "fermate", "sorgente"}

def json_is_complete(data: dict) -> bool:
    if not isinstance(data, dict) or "corse" not in data:
        log.warning("[JSON] Struttura mancante")
        return False
    if not data["corse"]:
        log.warning("[JSON] 'corse' è vuota")
        return False
    for i, c in enumerate(data["corse"]):
        miss = REQUIRED - set(c.keys())
        if miss:
            log.warning(f"[JSON] Record {i} manca: {miss}")
            return False
    return True


def build_json() -> dict:
    log.info("[JSON] ── Costruzione da PDF nelle sottocartelle...")
    if not PDF_DIR.exists():
        log.warning(f"[JSON] Cartella pdf/ non trovata → creazione")
        PDF_DIR.mkdir(parents=True, exist_ok=True)

    all_corse: list[dict] = []
    
    # Definisci le sottocartelle e i loro tipi corrispondenti
    subdir_mapping = [
        ("urbani", "urbano"),
        ("extraurbani", "extraurbano"),
        ("treni", "treno")
    ]
    
    for subdir, tipo in subdir_mapping:
        subdir_path = PDF_DIR / subdir
        if not subdir_path.exists():
            log.warning(f"[JSON] Subdirectory {subdir} non trovata in {PDF_DIR}")
            continue
            
        pdfs = list(subdir_path.glob("*.pdf"))
        log.info(f"[JSON] In {subdir}: {len(pdfs)} PDF trovati")
        
        for pdf_path in pdfs:
            all_corse.extend(parse_pdf(pdf_path, tipo_forzato=tipo))

    # --- CALCOLO CADENZA MEDIANA PER FERMATA ---
    log.info("[JSON] Calcolo cadenza mediana per ogni fermata...")
    
    # Dizionario per accumulare tutti gli orari per ogni (linea, direzione, fermata)
    stop_times = defaultdict(list)
    
    # Prima passata: raccogli tutti gli orari
    for record in all_corse:
        linea = record.get('linea')
        direzione = record.get('direzione')
        for fermata_info in record.get('fermate', []):
            fermata = fermata_info.get('fermata')
            orario = fermata_info.get('orario')
            if linea and direzione and fermata and orario:
                key = (linea, direzione, fermata)
                stop_times[key].append(orario)
    
    # Dizionario per memorizzare la cadenza calcolata per ogni key
    cadenza_map = {}
    
    # Seconda passata: calcola la mediana degli intervalli per ogni key
    for key, times_list in stop_times.items():
        if len(times_list) < 2:
            cadenza_map[key] = None  # Non abbastanza dati per calcolare cadenza
            continue
        
        # Funzione di conversione da "HH:MM" a minuti totali
        def to_minutes(time_str):
            try:
                h, m = map(int, time_str.split(':'))
                return h * 60 + m
            except ValueError:
                return None
        
        # Converti tutti gli orari in minuti e filtra eventuali errori
        minutes_list = []
        for t in times_list:
            m = to_minutes(t)
            if m is not None:
                minutes_list.append(m)
        
        # Se non abbiamo abbastanza valori validi, imposta cadenza a None
        if len(minutes_list) < 2:
            cadenza_map[key] = None
            continue
        
        # Ordina i minuti
        minutes_list.sort()
        
        # Calcola gli intervalli tra partenze consecutive
        intervals = []
        for i in range(1, len(minutes_list)):
            interval = minutes_list[i] - minutes_list[i-1]
            # Ignora intervalli nulli o negativi (dovrebbero non esserci dopo lo sort)
            if interval > 0:
                intervals.append(interval)
        
        # Se ci sono intervalli validi, calcola la mediana
        if intervals:
            cadenza_map[key] = statistics.median(intervals)
        else:
            cadenza_map[key] = None
    
    # Terza passata: aggiungi la cadenza a ogni fermata in ogni record
    for record in all_corse:
        linea = record.get('linea')
        direzione = record.get('direzione')
        for fermata_info in record.get('fermate', []):
            fermata = fermata_info.get('fermata')
            if linea and direzione and fermata:
                key = (linea, direzione, fermata)
                fermata_info['cadenza'] = cadenza_map.get(key)
            else:
                fermata_info['cadenza'] = None

    data = {
        "generato_il":    datetime.now().isoformat(timespec="seconds"),
        "totale_corse":   len(all_corse),
        "pdf_analizzati": [p.name for p in PDF_DIR.rglob("*.pdf")],  # Tutti i PDF trovati
        "corse":          all_corse,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"[JSON] ✅ Salvato: {JSON_PATH}  ({len(all_corse)} corse)")
    return data


def load_or_build_json() -> dict:
    if JSON_PATH.exists():
        log.info(f"[JSON] File trovato: {JSON_PATH}")
        try:
            with open(JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if json_is_complete(data):
                log.info("[JSON] Completo. Caricato.")
                return data
            log.warning("[JSON] Incompleto → ricostruzione")
        except json.JSONDecodeError as e:
            log.error(f"[JSON] Corrotto ({e}) → ricostruzione")
    else:
        log.info("[JSON] Non trovato → costruzione da zero")
    return build_json()


# ═══════════════════════════════════════════════
# STATO GLOBALE
# ═══════════════════════════════════════════════
DB: dict = {}


def build_system_prompt() -> str:
    """
    Costruisce il system prompt con TUTTE le fermate e orari,
    nessun troncamento, forzando la risposta in italiano.
    Include struttura dati chiara e esempio per migliorare la comprensione del modello.
    """
    corse = DB.get("corse", [])
    righe = []
    for c in corse:
        fermate_str = " → ".join(
            f"{f['fermata']} ({f['orario']})" + 
            (f" - Ogni {int(f['cadenza'])} min" if f.get('cadenza') is not None else "")
            for f in c.get("fermate", [])
        )
        note_str = " | ".join(c.get("note", []))
        ctx_str  = " | ".join(c.get("contesto", [])[:5])
        riga = (
            f"[{c['tipo'].upper()}] Linea {c['linea']} "
            f"Dir: {c['direzione']} "
            f"Fermate: {fermate_str or 'N/D'}"
        )
        if note_str:
            riga += f" | Note: {note_str}"
        if ctx_str:
            riga += f" | Info: {ctx_str}"
        righe.append(riga)

    orari = "\n".join(righe) if righe else "Nessun dato disponibile."

    return (
        "Sei un assistente virtuale per il Trasporto Pubblico Locale "
        "del Friuli Venezia Giulia (TPL FVG).\n\n"
        "LINGUA: Rispondi SEMPRE e SOLO in italiano. "
        "Non usare mai l'inglese o altre lingue, qualunque sia la lingua della domanda. "
        "Ogni tua risposta deve essere in italiano corretto.\n\n"
        "DATI: Usa esclusivamente i dati riportati qui sotto. "
        "Se un'informazione non è presente, dillo chiaramente in italiano. "
        "Non inventare orari o fermate.\n\n"
        "STRUTTURA DEI DATI:\n"
        "Ogni linea rappresenta una corsa specifica con formato:\n"
        "[TIPO] Linea NUMERO Linea Dir: DESTINAZIONE Fermate: FERMA1 (ORARIO1) - Ogni C1 min → FERMA2 (ORARIO2) - Ogni C2 min → ...\n"
        "dove:\n"
        "  • TIPO: TRENO, URBANO, EXTRAURBANO o AUTOBUS\n"
        "  • NUMERO Linea: identificatore (es. 10, 51, 100)\n"
        "  • DESTINAZIONE: direzione/capolinea\n"
        "  • Per ogni fermata: FERMA (ORARIO) - Ogni C min\n"
        "    * ORARIO: formato HH:MM\n"
        "    * C: cadenza mediana in minuti (null se non calcolabile)\n"
        "  • NOTE: avvisi o informazioni aggiuntive\n"
        "  • INFO: contesto estratto dal PDF\n\n"
        "COMPLETEZZA: Fornisci risposte esaustive con tutti gli orari pertinenti. "
        "Se esistono più corse/direzioni, elencale tutte.\n\n"
        "ESEMPIO DI RISPOSTA CORRETTA:\n"
        "Domanda: \"A che ora passa la linea 10 in direzione Piazza Unità alla fermata Repubblica?\"\n"
        "Risposta: \"Secondo gli orari forniti, la linea 10 in direzione Piazza Unità ferma alla fermata Repubblica alle 08:15, 08:30 e 08:45. "
        "La cadenza mediana è di 15 minuti tra una partenza e l'altra.\"\n\n"
        f"=== DATI ORARI TPL FVG (aggiornati al {DB.get('generato_il','N/D')}) ===\n"
        f"{orari}\n"
        "=== FINE DATI ===\n\n"
        "Rispondi SOLO in italiano."
    )


# ═══════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB
    log.info("═" * 55)
    log.info("  🚌  Mini AI TPL FVG — Avvio")
    log.info("═" * 55)
    DB = load_or_build_json()
    log.info(f"[APP] Pronto. Corse: {DB.get('totale_corse', 0)}")
    yield
    log.info("[APP] Shutdown.")

app = FastAPI(title="Mini AI TPL FVG", lifespan=lifespan)


@app.get("/api/status")
async def api_status():
    corse = DB.get("corse", [])
    tipi: dict[str, int] = {}
    for c in corse:
        t = c.get("tipo", "N/D")
        tipi[t] = tipi.get(t, 0) + 1
    
    # Nuova struttura per le linee: lista di oggetti {linea, tipo} senza duplicati
    linee_set = set()
    for c in corse:
        linea = c.get('linea')
        tipo = c.get('tipo')
        if linea and tipo:
            linee_set.add((linea, tipo))
    
    linee_dettaglio = [{"linea": linea, "tipo": tipo} for linea, tipo in linee_set]
    linee_dettaglio.sort(key=lambda x: (x['linea'], x['tipo']))  # Ordinato per linea poi tipo
    
    return {
        "generato_il":    DB.get("generato_il"),
        "totale_corse":   len(corse),
        "pdf_analizzati": DB.get("pdf_analizzati", []),
        "tipi":           tipi,
        "linee":          linee_dettaglio,  # Nuova struttura: lista di oggetti
    }


@app.post("/api/reload")
async def api_reload():
    global DB
    log.info("[API] Ricaricamento richiesto")
    if JSON_PATH.exists():
        JSON_PATH.unlink()
    DB = build_json()
    return {"ok": True, "totale_corse": DB.get("totale_corse", 0)}


@app.post("/api/chat")
async def api_chat(req: Request):
    body     = await req.json()
    history  = body.get("history", [])
    user_msg = body.get("message", "").strip()
    log.debug(f"[CHAT] Messaggio: {user_msg[:120]}")

    if not user_msg:
        return JSONResponse({"error": "Messaggio vuoto"}, status_code=400)

    system_prompt = build_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": f"[Rispondi in italiano] {user_msg}"})

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   True,
        "options":  {
            "temperature": 0.2,
            "num_predict": 2048,   # risposte lunghe e complete
        },
    }

    async def stream_tokens() -> AsyncGenerator[str, None]:
        log.debug("[OLLAMA] Avvio streaming...")
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        log.error(f"[OLLAMA] Errore {resp.status_code}: {err[:200]}")
                        yield f"data: {json.dumps({'token': f'[Errore Ollama {resp.status_code}]'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if chunk.get("done"):
                                log.debug("[OLLAMA] Stream completato")
                                yield "data: [DONE]\n\n"
                                return
                        except json.JSONDecodeError:
                            log.warning(f"[OLLAMA] Chunk non JSON: {line[:60]}")
        except httpx.ConnectError:
            log.error("[OLLAMA] Connessione rifiutata")
            yield f"data: {json.dumps({'token': '[Errore: Ollama non raggiungibile. Avvia ollama serve]'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.error(f"[OLLAMA] Errore: {e}", exc_info=True)
            yield f"data: {json.dumps({'token': f'[Errore: {e}]'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


# ═══════════════════════════════════════════════
# SERVIZIO FILE STATICI (HTML FRONTEND)
# ═══════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    """Servisce il file HTML frontend"""
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    module_name = Path(__file__).stem
    log.info(f"Avvio: uvicorn {module_name}:app  →  http://127.0.0.1:8000")
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
