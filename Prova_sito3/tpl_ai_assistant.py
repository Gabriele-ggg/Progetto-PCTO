#!/usr/bin/env python3
"""
Mini AI TPL FVG — Backend FastAPI + Uvicorn
PDF letti da ./pdf/urbani/, ./pdf/extraurbani/, ./pdf/treni/
Ollama locale per risposte strutturate.
Nel JSON gli orari sono riportati come ore (formato ["07","08","09",…]).
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
# COSTANTI E REGEX
# ═══════════════════════════════════════════════
SCRIPT_DIR   = Path(__file__).parent.resolve()
PDF_DIR      = SCRIPT_DIR / "pdf"
JSON_PATH    = SCRIPT_DIR / "tpl_fvg_orari.json"
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "granite4:3b"

TRENO_KW  = ["treno","train","intercity","freccia","italo","regionale","trenitalia","fs ","ferroviaria","rail"]
EXTRA_KW  = ["extraurbano","extra-urbano","interurbano","pullman","corriera","extraurb","suburbano"]
URBANO_KW = ["urbano","urban","città","city","bus u","linea u","apu","atap","apt trieste"]

# Gruppo non-capturing (?:...) per evitare che re.findall tronchi i minuti e supporto al punto (.)
TIME_RE   = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b")
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

TEXT_LENGTH_THRESHOLD = 30000

# ═══════════════════════════════════════════════
# FUNZIONI DI SUPPORTO DIZIONARIO
# ═══════════════════════════════════════════════

def detect_type_from_text(text: str) -> str:
    tl = text.lower()
    for kw in TRENO_KW:
        if kw in tl: return "treno"
    for kw in EXTRA_KW:
        if kw in tl: return "extraurbano"
    for kw in URBANO_KW:
        if kw in tl: return "urbano"
    return "autobus"


def estrai_ora(t: str) -> str:
    t_pulito = t.replace(".", ":")
    parts = t_pulito.split(":")
    if len(parts) != 2:
        return "00"
    try:
        hour = int(parts[0])
        return f"{hour:02d}"
    except ValueError:
        return "00"


def estrai_fermate_e_orari(testo: str) -> List[Dict[str, str]]:
    ferma_orario: List[Dict[str, str]] = []
    for line in testo.splitlines():
        line = line.strip()
        if not line:
            continue
        orari = TIME_RE.findall(line)
        if not orari:
            continue
        nome = TIME_RE.sub("", line).strip(" |•-–—\t:/.")
        nome = re.sub(r"\s{2,}", " ", nome).strip()
        if len(nome) < 2 or re.fullmatch(r"[\d\s\.\-\|/]+", nome):
            continue
        for o in orari:
            ferma_orario.append({"fermata": nome, "orario": estrai_ora(o)})
    return ferma_orario


def estrai_blocchi_testo(testo: str) -> List[str]:
    blocchi: List[str] = []
    for line in testo.splitlines():
        pulita = line.strip()
        if len(pulita.split()) >= 4 and not TIME_RE.search(pulita):
            blocchi.append(pulita)
    return blocchi[:20]  # <--- RISOLTO IL SYNTAX ERROR QUI!


def _estrai_testo_pagina(page) -> str:
    """Invertito l'ordine delle strategie dando la priorità assoluta al layout tabellare."""
    try:
        txt = page.extract_text(layout=True)
        if txt and txt.strip():
            return txt
    except Exception:
        pass
    try:
        txt = page.extract_text()
        if txt and txt.strip():
            return txt
    except Exception:
        pass
    try:
        parole = page.extract_words()
        if parole:
            return " ".join(p["text"] for p in parole)
    except Exception:
        pass
    return ""

# ═══════════════════════════════════════════════
# PARSING DI UN PDF
# ═══════════════════════════════════════════════

def parse_pdf(pdf_path: Path, tipo_forzato: str | None = None) -> List[Dict]:
    log.debug(f"[PDF] Apertura: {pdf_path.name}")
    risultati: List[Dict] = []
    try:
        reader = PdfReader(str(pdf_path))
        n_pagine = len(reader.pages)
        log.debug(f"[PDF] {pdf_path.name} → {n_pagine} pagine")

        global_parts = []
        for i in range(min(5, n_pagine)):
            txt = _estrai_testo_pagina(reader.pages[i])
            global_parts.append(txt)
        full_text = "\n".join(global_parts)

        tipo_mezzo = tipo_forzato if tipo_forzato is not None else detect_type_from_text(full_text)

        m = LINEA_RE.search(full_text)
        linea_globale = m.group(1).upper().strip() if m else None
        if not linea_globale:
            fn_match = re.search(r"(?:linea|line|bus)?\s*([A-Za-z0-9/\-]{1,6})", pdf_path.stem, re.IGNORECASE)
            linea_globale = fn_match.group(1).upper().strip() if fn_match else pdf_path.stem.upper()

        m = DIR_RE.search(full_text)
        dir_globale = m.group(1).strip() if m else None
        if not dir_globale:
            dir_match = re.search(r"(?:direzione|dir|verso|per)[:\-_ ]\s*([A-Za-z0-9 ]{3,50})", pdf_path.stem, re.IGNORECASE)
            dir_globale = dir_match.group(1).strip() if dir_match else pdf_path.stem.replace('_', ' ').replace('-', ' ').strip()

        note_globali = {mn.group(1).strip() for mn in NOTE_RE.finditer(full_text)}

        # Aggregatore ad albero per evitare record duplicati per singola fermata
        agg_corse: Dict[Tuple[str, str], Dict] = {}
        pagina_riferimento: int | None = None

        for idx, page in enumerate(reader.pages, start=1):
            testo = _estrai_testo_pagina(page)
            if len(testo.strip()) < TEXT_LENGTH_THRESHOLD: # <--- RISOLTO IL BUG DI TXT INDEFINITO QUI!
                log.debug(f"[PDF] Pagina {idx}: testo scarso")
            if not testo.strip():
                continue

            m = LINEA_RE.search(testo)
            linea = m.group(1).upper().strip() if m else linea_globale
            m = DIR_RE.search(testo)
            direzione = m.group(1).strip() if m else dir_globale

            if not linea or not direzione:
                continue

            ferme_pagina = estrai_fermate_e_orari(testo)
            if not ferme_pagina:
                continue

            if pagina_riferimento is None:
                pagina_riferimento = idx

            chiave = (linea, direzione)
            if chiave not in agg_corse:
                agg_corse[chiave] = {
                    "fermate_ordine": [],
                    "fermate_orari": defaultdict(set),
                    "note": set(note_globali),
                    "contesto": []
                }

            agg_corse[chiave]["note"].update([mn.group(1).strip() for mn in NOTE_RE.finditer(testo)])
            agg_corse[chiave]["contesto"].extend(estrai_blocchi_testo(testo))

            for f in ferme_pagina:
                f_nome = f["fermata"]
                ora = f["orario"]
                if f_nome not in agg_corse[chiave]["fermate_ordine"]:
                    agg_corse[chiave]["fermate_ordine"].append(f_nome)
                agg_corse[chiave]["fermate_orari"][f_nome].add(ora)

        for (linea, direzione), info in agg_corse.items():
            fermate_strutturate = []
            for f_nome in info["fermate_ordine"]:
                ore_unici = sorted(list(info["fermate_orari"][f_nome]))
                fermate_strutturate.append({
                    "fermata": f_nome,
                    "orario": ore_unici
                })

            record = {
                "tipo":      tipo_mezzo,
                "linea":     linea,
                "direzione": direzione,
                "fermate":   fermate_strutturate,
                "note":      sorted(list(info["note"])),
                "contesto":  list(dict.fromkeys(info["contesto"])),
                "sorgente":  pdf_path.name,
                "pagina":    pagina_riferimento if pagina_riferimento is not None else 1,
            }
            risultati.append(record)

        log.info(f"[PDF] ✅ {pdf_path.name}: generati {len(risultati)} percorsi integrati.")
    except Exception as e:
        log.error(f"[PDF] ❌ Errore su {pdf_path.name}: {e}", exc_info=True)
    return risultati


# ═══════════════════════════════════════════════
# GESTIONE JSON
# ═══════════════════════════════════════════════
REQUIRED = {"tipo", "linea", "direzione", "fermate", "sorgente"}

def json_is_complete(data: dict) -> bool:
    if not isinstance(data, dict) or "corse" not in data or not data["corse"]:
        return False
    for c in data["corse"]:
        if not REQUIRED.issubset(c.keys()) or not isinstance(c["fermate"], list):
            return False
        for f in c["fermate"]:
            if not isinstance(f, dict) or "fermata" not in f or "orario" not in f:
                return False
    return True


def build_json() -> dict:
    log.info("[JSON] ── Generazione database orari...")
    if not PDF_DIR.exists():
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
            continue
        pdfs = list(subdir_path.glob("*.pdf"))
        for pdf_path in pdfs:
            all_corse.extend(parse_pdf(pdf_path, tipo_forzato=tipo))

    data = {
        "generato_il":    datetime.now().isoformat(timespec="seconds"),
        "totale_corse":   len(all_corse),
        "pdf_analizzati": [p.name for p in PDF_DIR.rglob("*.pdf")],
        "corse":          all_corse,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"[JSON] ✅ Database salvato in: {JSON_PATH}")
    return data


def load_or_build_json() -> dict:
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if json_is_complete(data):
                log.info("[JSON] Caricato con successo.")
                return data
        except Exception:
            pass
    return build_json()


# ═══════════════════════════════════════════════
# SYSTEM PROMPT CACHE
# ═══════════════════════════════════════════════
DB: dict = {}
_SYSTEM_PROMPT_CACHE: str | None = None


def build_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE

    corse = DB.get("corse", [])
    righe = []
    for c in corse:
        fermate_parts = []
        for f in c.get("fermate", []):
            ferma = f.get("fermata", "")
            orari_str = ", ".join(f.get("orario", []))
            fermate_parts.append(f"{ferma} (Ore: {orari_str})")
        fermate_str = " → ".join(fermate_parts) if fermate_parts else "N/D"

        note_str = " | ".join(c.get("note", []))
        ctx_str  = " | ".join(c.get("contesto", [])[:3])
        
        riga = f"[{c['tipo'].upper()}] Linea {c['linea']} | Dir: {c['direzione']} | Percorso: {fermate_str}"
        if note_str: riga += f" | Note: {note_str}"
        if ctx_str:  riga += f" | Info: {ctx_str}"
        righe.append(riga)

    orari = "\n".join(righe) if righe else "Nessun dato disponibile."

    prompt = (
        "Sei un assistente virtuale per il Trasporto Pubblico Locale del Friuli Venezia Giulia (TPL FFVG).\n\n"
        "LINGUA: Rispondi SEMPRE e SOLO in italiano.\n\n"
        "DATI: Usa esclusivamente la sequenza lineare dei dati riportati qui sotto per calcolare i percorsi. "
        "Non inventare alcuna fermata o orario.\n\n"
        "=== DATI ORARI TPL FVG ===\n"
        f"{orari}\n"
        "=== FINE DATI ===\n"
    )
    
    _SYSTEM_PROMPT_CACHE = prompt # <--- RISOLTO IL CODICE UNREACHABLE QUI!
    log.info("[PROMPT] System prompt aggiornato in cache.")
    return prompt


# ═══════════════════════════════════════════════
# FASTAPI APIS
# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB, _SYSTEM_PROMPT_CACHE
    log.info(" 🚌 Mini AI TPL FVG — Inizializzazione database...")
    DB = load_or_build_json()
    _SYSTEM_PROMPT_CACHE = None
    yield

app = FastAPI(title="Mini AI TPL FVG", lifespan=lifespan)


@app.get("/api/status")
async def api_status():
    corse = DB.get("corse", [])
    tipi: dict[str, int] = {}
    linee_set = set()
    for c in corse:
        t = c.get("tipo", "N/D")
        tipi[t] = tipi.get(t, 0) + 1
        if c.get('linea') and c.get('tipo'):
            linee_set.add((c.get('linea'), c.get('tipo')))

    linee_dettaglio = [{"linea": l, "tipo": t} for l, t in linee_set]
    linee_dettaglio.sort(key=lambda x: (x['linea'], x['tipo']))

    return {
        "generato_il":    DB.get("generato_il"),
        "totale_corse":   len(corse),
        "pdf_analizzati": DB.get("pdf_analizzati", []),
        "tipi":           tipi,
        "linee":          linee_dettaglio,
    }


@app.post("/api/reload")
async def api_reload():
    global DB, _SYSTEM_PROMPT_CACHE
    if JSON_PATH.exists():
        JSON_PATH.unlink()
    DB = build_json()
    _SYSTEM_PROMPT_CACHE = None
    return {"ok": True, "totale_corse": DB.get("totale_corse", 0)}


@app.post("/api/chat")
async def api_chat(req: Request):
    body     = await req.json()
    history  = body.get("history", [])
    user_msg = body.get("message", "").strip()

    if not user_msg:
        return JSONResponse({"error": "Messaggio vuoto"}, status_code=400)

    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   True,
        "options":  {"temperature": 0.2, "num_predict": 1024},
    }

    async def stream_tokens() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'token': '[Errore Ollama]'})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line: continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if chunk.get("done"):
                            return
        except Exception as e:
            yield f"data: {json.dumps({'token': f'[Errore: {e}]'})}\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = SCRIPT_DIR / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("Frontend index.html non trovato.", status_code=404)


if __name__ == "__main__":
    import uvicorn
    module_name = Path(__file__).stem
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")