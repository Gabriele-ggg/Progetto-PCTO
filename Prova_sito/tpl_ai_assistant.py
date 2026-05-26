#!/usr/bin/env python3
"""
Mini AI TPL FVG — Backend FastAPI + Uvicorn
PDF letti da ./pdf/urbani/, ./pdf/extraurbani/, ./pdf/treni/
Ollama llama3:8b come LLM locale.
Nel JSON gli orari sono riportati solo come ore (es. ["07","08","09",…]).
"""

import os, sys, json, re, logging, httpx
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, List, Dict, Tuple
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

# Parole chiave per il riconoscimento del tipo (fallback, non usato se forzato dalla cartella)
TRENO_KW  = ["treno","train","intercity","freccia","italo","regionale","trenitalia","fs ","ferroviaria","rail"]
EXTRA_KW  = ["extraurbano","extra-urbano","interurbano","pullman","corriera","extraurb","suburbano"]
URBANO_KW = ["urbano","urban","città","city","bus u","linea u","apu","atap","apt trieste"]

# Regex – catturiamo ora e minuti (anche se i minuti verranno ignorati)
TIME_RE   = re.compile(r"\b([01]?\d|2[0-3]):[0-5]?\d\b")          # accetta anche 6:5
LINEA_RE  = re.compile(
    r"(?:linea|line|bus|autobus|treno|corsa|servizio|n[°\.]?)\s*[:\-]?\s*([A-Za-z0-9/\-]{1,6})",
    re.IGNORECASE,
)
DIR_RE    = re.compile(
    r"(?:direzione|dir\.?|verso|capolinea|destinazione|per|da\s+\S+\s+a)\s*[:\-]?\s*([^\n\r,;]{3,80})",
    re.IGNORECASE,
)
NOTE_RE   = re.compile(
    r"(?:note|avviso|info|attenzione|orario\s+estivo|orario\s+invernale|festivo|feriale)[:\s]+([^\n\r]{5,120})",
    re.IGNORECASE,
)

# Soglia sotto la quale consideriamo il testo estratto troppo poco
# (usato solo per loggare; senza OCR non tenta alcun recupero)
TEXT_LENGTH_THRESHOLD = 30

# ═══════════════════════════════════════════════
# FUNZIONI DI SUPPORTO
# ═══════════════════════════════════════════════

def detect_type_from_text(text: str) -> str:
    """Riconosce il tipo dal testo (usato solo se non forzato dalla cartella)."""
    tl = text.lower()
    for kw in TRENO_KW:
        if kw in tl: return "treno"
    for kw in EXTRA_KW:
        if kw in tl: return "extraurbano"
    for kw in URBANO_KW:
        if kw in tl: return "urbano"
    return "autobus"


def estrai_ora(t: str) -> str:
    """
    Estrae l'ora da una stringa HH:MM (o H:MM) e la restituisce
    come stringa a due cifre con zero iniziale (es. "07").
    """
    parts = t.split(":")
    if len(parts) != 2:
        return "00"
    try:
        hour = int(parts[0])
        return f"{hour:02d}"
    except ValueError:
        return "00"


def estrai_fermate_e_orari(testo: str) -> List[Dict[str, str]]:
    """
    Estrae tutte le coppie (fermata, ora) da una stringa di testo.
    Restituisce lista di dict: [{fermata: str, orario: str}] dove
    orario contiene SOLO l'ora a due cifre (es. "07").
    """
    ferma_orario: List[Dict[str, str]] = []
    for line in testo.splitlines():
        line = line.strip()
        if not line:
            continue
        orari = TIME_RE.findall(line)
        if not orari:
            continue
        # Rimuovi gli orari per ottenere il nome della fermata
        nome = TIME_RE.sub("", line).strip(" |•-–—\t:/")
        nome = re.sub(r"\s{2,}", " ", nome).strip()
        if len(nome) < 2 or re.fullmatch(r"[\d\s\.\-\|/]+", nome):
            continue
        for o in orari:
            ferma_orario.append({"fermata": nome, "orario": estrai_ora(o)})
    return ferma_orario


def estrai_blocchi_testo(testo: str) -> List[str]:
    """Blocchi di testo significativi (almeno 4 parole) per contesto aggiuntivo."""
    blocchi: List[str] = []
    for line in testo.splitlines():
        pulita = line.strip()
        if len(pulita.split()) >= 4 and not TIME_RE.search(pulita):
            blocchi.append(pulita)
    return blocchi[:20]   # max 20 righe per pagina


def _estrai_testo_pagina(page) -> str:
    """Prova diverse strategie per estrarre il testo da una pagina (senza OCR)."""
    # 1. Estrattore standard
    txt = page.extract_text()
    if txt and txt.strip():
        return txt
    # 2. Con layout=True (utile per tabelle)
    try:
        txt = page.extract_text(layout=True)
        if txt and txt.strip():
            return txt
    except Exception:
        pass
    # 3. Parole singole
    try:
        parole = page.extract_words()
        if parole:
            txt = " ".join(p["text"] for p in parole)
            if txt.strip():
                return txt
    except Exception:
        pass
    return ""   # niente trovato


# ═══════════════════════════════════════════════
# PARSING DI UN PDF
# ═══════════════════════════════════════════════

def parse_pdf(pdf_path: Path, tipo_forzato: str | None = None) -> List[Dict]:
    """
    Legge il PDF e restituisce una lista di record.
    Ogni record corrisponde a una combinazione unica (linea, direzione, fermata)
    e contiene:
        - tipo
        - linea
        - direzione
        - ferme: lista di dict {"fermata": str, "orario": List[str]}   # ora a due cifre
        - note: lista di stringhe (deduplicate)
        - contesto: lista di stringhe (deduplicate, ordine preservato)
        - sorgente: nome file PDF
        - pagina: numero della pagina da cui proviene il record (prima pagina con dati)
    """
    log.debug(f"[PDF] Apertura: {pdf_path.name}")
    risultati: List[Dict] = []
    try:
        reader = PdfReader(str(pdf_path))
        n_pagine = len(reader.pages)
        log.debug(f"[PDF] {pdf_path.name} → {n_pagine} pagine")

        # ---------- Analisi globale (prime 5 pagine) per tipo e metadati ----------
        global_parts = []
        for i in range(min(5, n_pagine)):
            txt = _estrai_testo_pagina(reader.pages[i])
            # Log utile se il testo è scarso (senza OCR non facciamo nulla di più)
            if len(txt.strip()) < TEXT_LENGTH_THRESHOLD:
                log.debug(f"[PDF] Pagina {i+1}: testo scarso ({len(txt.strip())} caratteri)")
            global_parts.append(txt)
        full_text = "\n".join(global_parts)
        log.debug(f"[PDF] Testo globale (prime 5 pagine): {repr(full_text[:200])}")

        # Tipo: forzato dalla cartella oppure rilevato dal testo
        if tipo_forzato is not None:
            tipo_mezzo = tipo_forzato
            log.debug(f"[PDF] Tipo forzato dalla cartella: {tipo_mezzo}")
        else:
            tipo_mezzo = detect_type_from_text(full_text)
            log.debug(f"[PDF] Tipo rilevato dal testo: {tipo_mezzo}")

        # Linea e direzione globali (fallback)
        m = LINEA_RE.search(full_text)
        linea_globale = m.group(1).upper().strip() if m else None
        m = DIR_RE.search(full_text)
        dir_globale = m.group(1).strip() if m else None
        log.debug(f"[PDF] Linea globale={linea_globale} Dir globale={dir_globale}")

        # Note globali
        note_globali: set = set()
        for mn in NOTE_RE.finditer(full_text):
            note_globali.add(mn.group(1).strip())
        log.debug(f"[PDF] Note globali: {note_globali}")

        # Strutture di aggregazione per l’intero PDF
        agg_stop: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)   # (linea, direzione, ferma) -> lista ore
        agg_note: set = set(note_globali)                                     # insieme di note
        agg_contesto: List[str] = []                                          # contesto raccolto
        pagina_riferimento: int | None = None                                 # prima pagina utile

        # ---------- Analisi pagina per pagina ----------
        for idx, page in enumerate(reader.pages, start=1):
            testo = _estrai_testo_pagina(page)
            if len(txt.strip()) < TEXT_LENGTH_THRESHOLD:
                log.debug(f"[PDF] Pagina {idx}: testo scarso ({len(txt.strip())} caratteri)")
            if not testo.strip():
                log.debug(f"[PDF] Pagina {idx}: vuota/scansione, skip")
                continue

            # Linea locale (se mancante usa globale)
            m = LINEA_RE.search(testo)
            linea = m.group(1).upper().strip() if m else linea_globale
            # Direzione locale
            m = DIR_RE.search(testo)
            direzione = m.group(1).strip() if m else dir_globale

            if not linea or not direzione:
                log.debug(f"[PDF] Pagina {idx}: linea/direzione mancanti, skip")
                continue

            # Fermate + orari (ora solo ora)
            ferme_pagina = estrai_fermate_e_orari(testo)
            log.debug(f"[PDF] Pagina {idx}: trovate {len(ferme_pagina)} fermate/orari")
            if not ferme_pagina:
                log.debug(f"[PDF] Pagina {idx}: nessuna fermata trovata, skip")
                continue

            # Prima pagina utile (per sorgente/pagina)
            if pagina_riferimento is None:
                pagina_riferimento = idx

            # Note locali
            note_pagina = [mn.group(1).strip() for mn in NOTE_RE.finditer(testo)]
            agg_note.update(note_pagina)
            log.debug(f"[PDF] Pagina {idx}: note trovate: {note_pagina}")

            # Contesto locale
            contesto_pagina = estrai_blocchi_testo(testo)
            agg_contesto.extend(contesto_pagina)
            log.debug(f"[PDF] Pagina {idx}: blocchi di contesto: {len(contesto_pagina)}")

            # Aggiorna le fermate/Orari
            for f in ferme_pagina:
                ferma_nome = f["fermata"]
                ora = f["orario"]
                chiave = (linea, direzione, ferma_nome)
                agg_stop[chiave].append(ora)

        # ---------- Se niente trovato, restituiamo lista vuota ----------
        if not agg_stop:
            log.info(f"[PDF] ⚠️ {pdf_path.name}: nessun dato estratto")
            return risultati

        # ---------- Costruzione dei record finali ----------
        for (linea, direzione, ferma_nome), ore_list in agg_stop.items():
            ore_unici = sorted(set(ore_list))
            record = {
                "tipo":      tipo_mezzo,
                "linea":     linea,
                "direzione": direzione,
                "fermate":   [
                    {"fermata": ferma_nome, "orario": ore_unici}
                ],
                "note":      list(agg_note),          # già deduplicato
                "contesto":  list(dict.fromkeys(agg_contesto)),  # deduplica preservando ordine
                "sorgente":  pdf_path.name,
                "pagina":    pagina_riferimento if pagina_riferimento is not None else 1,
            }
            risultati.append(record)
            log.debug(
                f"[PDF] Record: linea={linea}, dir={direzione}, "
                f"fermata={ferma_nome}, orari={ore_unici}"
            )

        log.info(f"[PDF] ✅ {pdf_path.name}: {len(risultati)} record estratti (fermate uniche)")
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
        ferme = c.get("fermate", [])
        if not isinstance(ferme, list):
            log.warning(f"[JSON] Record {i}: 'fermate' non è una lista")
            return False
        for j, f in enumerate(ferme):
            if not isinstance(f, dict) or "fermata" not in f or "orario" not in f:
                log.warning(f"[JSON] Record {i}, fermata {j}: struttura non valida")
                return False
            if not isinstance(f["orario"], list) or not all(isinstance(t, str) for t in f["orario"]):
                log.warning(f"[JSON] Record {i}, fermata {j}: 'orario' non è lista di stringhe")
                return False
    return True


def build_json() -> dict:
    log.info("[JSON] ── Costruzione da PDF nelle sottocartelle...")
    if not PDF_DIR.exists():
        log.warning(f"[JSON] Cartella pdf/ non trovata → creazione")
        PDF_DIR.mkdir(parents=True, exist_ok=True)

    all_corse: List[dict] = []
    subdir_mapping = [
        ("urbani", "urbano"),
        ("extraurbani", "extraurbano"),
        ("treni", "treno"),
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

    data = {
        "generato_il":    datetime.now().isoformat(timespec="seconds"),
        "totale_corse":   len(all_corse),
        "pdf_analizzati": [p.name for p in PDF_DIR.rglob("*.pdf")],  # tutti i PDF trovati
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
    Nel prompt indichiamo che gli orari sono solo le ore.
    """
    corse = DB.get("corse", [])
    righe = []
    for c in corse:
        fermate_parts = []
        for f in c.get("fermate", []):
            ferma = f.get("fermata", "")
            orari = f.get("orario", [])
            orari_str = ", ".join(orari)
            fermate_parts.append(f"{ferma} ({orari_str})")
        fermate_str = " → ".join(fermate_parts) if fermate_parts else "N/D"

        note_str = " | ".join(c.get("note", []))
        ctx_str  = " | ".join(c.get("contesto", [])[:5])
        riga = (
            f"[{c['tipo'].upper()}] Linea {c['linea']} "
            f"Dir: {c['direzione']} "
            f"Fermate: {fermate_str}"
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
        "[TIPO] Linea NUMERO Linea Dir: DESTINAZIONE Fermate: FERMA1 (ORARIO1, ORARIO2, ...) → FERMA2 (ORARIO3, ORARIO4, ...) → ...\n"
        "dove:\n"
        "  • TIPO: TRENO, URBANO, EXTRAURBANO o AUTOBUS\n"
        "  • NUMERO Linea: identificatore (es. 10, 51, 100)\n"
        "  • DESTINAZIONE: direzione/capolinea\n"
        "  • Per ogni fermata: FERMA (ORARIO1, ORARIO2, ...)\n"
        "    * ORARIO: formato HH (ora a due cifre, es. \"07\"), può essere più di un orario separato da virgola\n"
        "  • NOTE: avvisi o informazioni aggiuntive\n"
        "  • INFO: contesto estratto dal PDF\n\n"
        "COMPLETEZZA: Fornisci risposte esaustive con tutti gli orari pertinenti. "
        "Se esistono più corse/direzioni, elencale tutte.\n\n"
        "ESEMPIO DI RISPOSTA CORRETTA:\n"
        "Domanda: \"A che ora passa la linea 10 in direzione Piazza Unità alla fermata Repubblica?\"\n"
        "Risposta: \"Secondo gli orari forniti, la linea 10 in direzione Piazza Unità ferma alla fermata Repubblica alle 07, 08 e 09.\"\n\n"
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

    linee_set = set()
    for c in corse:
        linea = c.get('linea')
        tipo = c.get('tipo')
        if linea and tipo:
            linee_set.add((linea, tipo))

    linee_dettaglio = [{"linea": linea, "tipo": tipo} for linea, tipo in linee_set]
    linee_dettaglio.sort(key=lambda x: (x['linea'], x['tipo']))

    return {
        "generato_il":    DB.get("generato_il"),
        "totale_corse":   len(corse),
        "pdf_analizzati": DB.get("pdf_analizzati", []),
        "tipi":           tipi,
        "linee":          linee_dettaglio,   # lista di oggetti {linea, tipo}
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
