"""
rag_service.py
==============
Servizio RAG per trasporto pubblico locale.

Lo scraping web è abilitato SOLO se il sito pubblica un file robots.txt
raggiungibile.  Se il file è assente (risposta non-200 o errore di rete)
la scansione viene rifiutata automaticamente.  Se è presente, il suo
contenuto viene rispettato per l'agente configurato.
"""

import os
import json
import re
import gc
import urllib.robotparser
from datetime import date, datetime
from collections import deque
from urllib.parse import urlparse
import difflib

from fastapi import HTTPException

# ── Dipendenze opzionali ───────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

try:
    import chromadb
except Exception:
    chromadb = None

try:
    import requests as _requests
except Exception:
    _requests = None

try:
    from bs4 import BeautifulSoup as _BS4
except Exception:
    _BS4 = None

# Non inizializzare automaticamente qui: `initialize_system()` viene chiamato
# esplicitamente dal server API (`backend.api_server`) durante l'avvio.
DB_INSTANCE = None


# ─────────────────────────────────────────────────────────────────────────────
# PERCORSI E CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────
ROOT_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_SOURCE_ROOT  = os.path.join(ROOT_PROJECT_DIR, "pdf")
CHROMA_PATH      = os.path.join(ROOT_PROJECT_DIR, "data", "chroma_db")
EMBED_MODEL      = "nomic-embed-text:latest"

# Modello AI principale da usare per eseguire il compito.
# Cambialo qui, oppure sovrascrivilo con la variabile d'ambiente RAG_LLM_MODEL.
AI_MODEL_TO_USE: str = os.environ.get("RAG_LLM_MODEL", "granite4:3b").strip() or "granite4:3b"

# ── Configurazione scraping ───────────────────────────────────────────────
#
# SCRAPE_SITES  ←  MODIFICA QUI per aggiungere o rimuovere siti da scansionare.
#
# Ogni voce è un dizionario con i seguenti campi:
#
#   url         (obbligatorio) URL completo della pagina da recuperare.
#   description (facoltativo)  Descrizione leggibile, usata nei log.
#   enabled     (facoltativo)  False per disabilitare temporaneamente la voce
#                              senza cancellarla. Default: True.
#
# Esempio con più siti:
#
#   SCRAPE_SITES = [
#       {
#           "url":         "https://example.com/orari",
#           "description": "Orari autobus esempio",
#       },
#       {
#           "url":         "https://altro-sito.it/pagina",
#           "description": "Altro sito di prova",
#           "enabled":     False,   # ← commentato/disabilitato
#       },
#   ]
#
# Nota: lo scraping viene eseguito SOLO se il sito espone un robots.txt
# valido che consente l'accesso per l'agente SCRAPE_USER_AGENT.
# ─────────────────────────────────────────────────────────────────────────────
SCRAPE_SITES: list[dict] = [
    #{
    #    "url":         "",
    #    "description": "Sito di esempio (da configurare)",
    #},
    # Aggiungi altri siti qui ↓
    # {
    #     "url":         "https://altro-sito.it/pagina",
    #     "description": "Descrizione del sito",
    # },
]

# Derivati automaticamente da SCRAPE_SITES — non modificare direttamente.
SCRAPE_TARGET_URLS:   list[str] = [
    s["url"] for s in SCRAPE_SITES if s.get("enabled", True)
]
SCRAPE_ALLOWED_HOSTS: list[str] = [
    urlparse(s["url"]).netloc.split(":")[0]
    for s in SCRAPE_SITES if s.get("enabled", True)
]

# User-agent dichiarato nelle richieste HTTP e confrontato con robots.txt.
SCRAPE_USER_AGENT = "PCTO-Scraper/1.0"

# Modello LLM selezionato.
# Valori possibili:
#   "rule-based"                  → nessun LLM, solo logica a regole
#   "<nome_ollama>"               → modello Ollama locale (es. "mistral:7b")
#
# Può essere sovrascritto a runtime con set_model() oppure impostato tramite
# la variabile d'ambiente RAG_LLM_MODEL al lancio del server.
SELECTED_MODEL: str = AI_MODEL_TO_USE

# Riferimento globale a un'istanza ChatOllama caricata (solo per provider Ollama)
LOADED_LLM = None


def get_selected_model() -> str:
    try:
        return SELECTED_MODEL
    except Exception:
        return ""


def _init_model_from_env() -> None:
    """Inizializza LOADED_LLM se RAG_LLM_MODEL punta a un modello Ollama."""
    global LOADED_LLM, SELECTED_MODEL
    m = SELECTED_MODEL
    if not m:
        print(f"[WARN] Nessun modello selezionato (SELECTED_MODEL vuoto).")
        return
    if ChatOllama is None:
        print(f"[WARN] RAG_LLM_MODEL={m} ma ChatOllama non disponibile.")
        return
    try:
        LOADED_LLM = ChatOllama(model=m, temperature=0.0, top_p=1.0)
        print(f"[INFO] Modello Ollama '{m}' caricato da RAG_LLM_MODEL.")
    except Exception as exc:
        print(f"[WARN] Impossibile caricare '{m}' da RAG_LLM_MODEL: {exc}")


# Esegui all'import del modulo
_init_model_from_env()


def set_model(model_name: str) -> None:
    """
    Imposta il modello LLM da usare per tutte le risposte.

    Esempi:
        set_model("rule-based")                          # nessun LLM
        set_model("mistral:7b")                          # Ollama locale
    """
    global SELECTED_MODEL, LOADED_LLM, AI_MODEL_TO_USE
    model_name = (model_name or "").strip()
    print(f"[DEBUG] set_model called: '{model_name}'")

    AI_MODEL_TO_USE = model_name
    SELECTED_MODEL  = model_name

    if ChatOllama is None:
        print("[WARN] ChatOllama non disponibile; il modello è stato selezionato ma non è possibile caricarlo.")
        LOADED_LLM = None
        return
    try:
        LOADED_LLM     = ChatOllama(model=model_name, temperature=0.0, top_p=1.0)
        SELECTED_MODEL = model_name
        print(f"[INFO] Modello impostato: {model_name} (provider Ollama)")
    except Exception as exc:
        print(f"[ERROR] Impossibile caricare il modello Ollama '{model_name}': {exc}")
        SELECTED_MODEL = model_name
        print(f"[INFO] Modello '{model_name}' selezionato ma non disponibile al momento")


def _call_llm(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.0,
) -> dict | None:
    """
    Invia un messaggio al modello LLM selezionato e restituisce la risposta con metadati.

    Ritorna un dizionario: {"text": risposta, "tokens": numero_token_approssimato}
    o None se nessun LLM è configurato.
    """
    global SELECTED_MODEL, LOADED_LLM
    print(f"[DEBUG] _call_llm: SELECTED_MODEL='{SELECTED_MODEL}'")

    if not SELECTED_MODEL:
        raise Exception(
            "Nessun modello LLM selezionato. "
            "Imposta la variabile d'ambiente RAG_LLM_MODEL o seleziona un modello tramite /api/set-model."
        )

    if ChatOllama is None:
        raise Exception(
            "ChatOllama non disponibile: installa e configura il provider Ollama per caricare il modello LLM."
        )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as exc:
        print(f"[WARN] LangChain non disponibile: {exc}")
        return None

    llm = LOADED_LLM
    if llm is None:
        try:
            llm = ChatOllama(model=SELECTED_MODEL, temperature=temperature, top_p=1.0)
            LOADED_LLM = llm
        except Exception as exc:
            raise Exception(f"Impossibile creare istanza Ollama per '{SELECTED_MODEL}': {exc}")

    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        resp = llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)

        words  = len(text.split())
        chars  = len(text)
        tokens = max(words, max(1, chars // 4))

        return {"text": text, "tokens": tokens}
    except Exception as exc:
        raise


def _detect_hallucinations(
    llm_text: str,
    snippet_text: str | None,
    available_categories: list[str] | None = None,
    known_places: set | None = None,
) -> list[str]:
    """Semplice rilevatore euristico di allucinazioni.
    Restituisce una lista di problemi trovati; lista vuota => nessuna anomalia rilevata.
    """
    issues = []
    try:
        snippet = (snippet_text or "").lower()

        # Orari
        times         = set(_TIME_PAT.findall(llm_text))
        snippet_times = set(_TIME_PAT.findall(snippet))
        for t in times:
            if t not in snippet_times:
                issues.append(f"missing_time:{t}")

        # Linee
        linea_matches = re.findall(r"\blinea\s+([0-9]{1,3}[A-Za-z]?)\b", llm_text, flags=re.I)
        for ln in linea_matches:
            if str(ln).lower() not in snippet:
                issues.append(f"missing_line:{ln}")

        # Categorie proibite
        text_low = llm_text.lower()
        if re.search(r"\btreno|treni\b", text_low):
            cats = [c.lower() for c in (available_categories or [])]
            if "treni" not in cats:
                issues.append("missing_category:treni")
        if re.search(r"\b(auto|macchina|automobile|in auto)\b", text_low):
            issues.append("forbidden_category:private_vehicle")

        # Luoghi sconosciuti
        if known_places is not None:
            tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'\s]{2,40}", llm_text)
            for tok in tokens:
                t = tok.strip()
                if not t:
                    continue
                if re.search(r"[A-ZÀ-ÖØ-Þ]", t):
                    low = t.lower()
                    if low not in known_places and len(low) > 2:
                        issues.append(f"unknown_place:{t}")

    except Exception as exc:
        print(f"[WARN] _detect_hallucinations error: {exc}")
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# ESTRAZIONE TESTO DA HTML
# ─────────────────────────────────────────────────────────────────────────────
_BLOCK_TAGS = {
    "script", "style", "noscript", "nav", "footer",
    "header", "aside", "iframe", "svg", "form",
}


def _extract_text_from_html(html: str) -> str:
    """Estrae testo significativo dall'HTML."""
    if _BS4 is not None:
        try:
            soup = _BS4(html, "html.parser")
            for tag in soup(_BLOCK_TAGS):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()
        except Exception:
            pass

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_vector_store(pdf_folder: str):
    """Inizializza (se possibile) ChromaDB e genera il file trasporti.json."""
    print("\n--- INIZIALIZZAZIONE SISTEMA: Generazione Dati ---")
    try:
        generate_transport_json(pdf_folder)
    except Exception as exc:
        print(f"[ERROR] Fallimento generazione dati: {exc}")
        return None

    if chromadb is None:
        print("[WARN] ChromaDB non disponibile, salto inizializzazione DB vettoriale.")
        return None

    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("my_documents")
        print("[OK] ChromaDB pronta.")
        return collection
    except Exception as exc:
        print(f"[ERROR] Errore inizializzazione ChromaDB: {exc}")
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def pdf_content(pdf_bytes: bytes) -> str:
    """Estrae testo da PDF usando PyPDFLoader, se disponibile."""
    if PyPDFLoader is None:
        return ""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs   = loader.load()
            return "\n\n[PAGINA]\n\n".join(d.page_content for d in docs)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as exc:
        print(f"[ERROR] pdf_content: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# SCHOOL CIRCULARS – CARICAMENTO IN MEMORIA
# ─────────────────────────────────────────────────────────────────────────────
SCHOOL_CIRCULARS_CACHE: dict[str, str] = {}
CIRCULARS_LOADED = False


def load_school_circulars_simple() -> None:
    """
    Carica i PDF delle circolari scolastiche dalla cartella pdf/circolari_scuola
    in memoria per ricerca testuale semplice (senza ChromaDB).
    """
    global SCHOOL_CIRCULARS_CACHE, CIRCULARS_LOADED

    if CIRCULARS_LOADED:
        return

    try:
        circulars_folder = os.path.join(PDF_SOURCE_ROOT, "circolari_scuola")

        if not os.path.exists(circulars_folder):
            print(f"[INFO] Cartella circolari non trovata: {circulars_folder}")
            CIRCULARS_LOADED = True
            return

        pdf_files = [f for f in os.listdir(circulars_folder) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"[INFO] Nessun PDF trovato nella cartella circolari: {circulars_folder}")
            CIRCULARS_LOADED = True
            return

        print(f"[INFO] Caricamento {len(pdf_files)} circolari scolastiche...")

        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(circulars_folder, pdf_file)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                content = pdf_content(pdf_bytes)
                if not content.strip():
                    print(f"[WARN] Contenuto vuoto dalla circolare: {pdf_file}")
                    continue
                SCHOOL_CIRCULARS_CACHE[pdf_file] = content
                print(f"[OK] Circolare caricata: {pdf_file}")
            except Exception as e:
                print(f"[ERROR] Errore caricamento circolare {pdf_file}: {e}")

        CIRCULARS_LOADED = True
        print(f"[OK] {len(SCHOOL_CIRCULARS_CACHE)} circolari scolastiche caricate in memoria")

    except Exception as e:
        print(f"[ERROR] Errore nel caricamento circolari: {e}")
        CIRCULARS_LOADED = True


# ─────────────────────────────────────────────────────────────────────────────
# SCHOOL CIRCULARS – RICERCA
# ─────────────────────────────────────────────────────────────────────────────
def search_school_circulars(query: str, max_results: int = 3) -> str:
    """
    Cerca le circolari scolastiche rilevanti per la query.
    Tenta prima una ricerca semantica su ChromaDB; se non disponibile,
    esegue una ricerca testuale sulle circolari caricate in memoria.
    Ritorna una stringa formattata con i risultati, o stringa vuota se assente.
    """
    # ── Tentativo 1: ChromaDB (ricerca semantica) ─────────────────────────────
    if chromadb is not None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                collection = client.get_collection("school_circulars")
            except Exception:
                print("[INFO] Collezione 'school_circulars' non trovata in ChromaDB.")
                collection = None

            if collection is not None:
                results = collection.query(
                    query_texts=[query],
                    n_results=max_results,
                )
                if results and results.get("documents") and results["documents"][0]:
                    formatted = "CIRCOLARI SCOLASTICHE RILEVANTI:\n"
                    for i, doc in enumerate(results["documents"][0], 1):
                        meta   = results["metadatas"][0][i - 1] if results.get("metadatas") else {}
                        source = meta.get("source", "Sconosciuta")
                        formatted += f"\n[{i}] Da: {source}\n{doc[:500]}...\n"
                    return formatted
        except Exception as e:
            print(f"[WARN] Ricerca circolari su ChromaDB fallita: {e}")

    # ── Tentativo 2: ricerca testuale in memoria ──────────────────────────────
    try:
        load_school_circulars_simple()

        if not SCHOOL_CIRCULARS_CACHE:
            return ""

        query_lower = query.lower()
        results     = []

        for filename, content in SCHOOL_CIRCULARS_CACHE.items():
            paragraphs = content.split("\n\n")
            matching   = []
            for para in paragraphs:
                if any(word in para.lower() for word in query_lower.split() if len(word) > 2):
                    matching.append(para.strip())
            if matching:
                for para in matching[:2]:
                    results.append((filename, para[:500]))

        if not results:
            return ""

        formatted = "CIRCOLARI SCOLASTICHE RILEVANTI:\n"
        for i, (filename, text) in enumerate(results[:max_results], 1):
            formatted += f"\n[{i}] Da: {filename}\n{text}...\n"
        return formatted

    except Exception as e:
        print(f"[ERROR] Errore ricerca circolari (testo): {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN / NOISE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_TIME_PAT  = re.compile(r"\b(\d{1,2}:\d{2})\b")
_DIGIT_PAT = re.compile(r"^[\d\s:.,;|\-\/\(\)\[\]]+$")

_NOISE_KEYWORDS = {
    "pagina", "pag.", "orario", "valido", "feriale", "festivo",
    "servizio", "servizio urbano", "specificato", "specifcato",
    "tpl", "autobus", "bus", "pullman", "treno", "ferrovia",
    "informazioni", "azienda", "note", "legenda", "leggenda",
    "dal", "al", "fino", "annuale", "invernale", "estivo",
    "lun", "mar", "mer", "gio", "ven", "sab", "dom",
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
}

_NOISE_PATTERNS = [
    re.compile(r"^\s*pagina\s*\d+\s*$",               re.I),
    re.compile(r"^\s*pag\.?\s*\d+\s*(/\s*\d+)?\s*$",  re.I),
    re.compile(r"^\s*[-_=*]{3,}\s*$"),
    re.compile(r"^\s*\[pagina\]\s*$",                  re.I),
    re.compile(r"^\s*(andata|ritorno)\s*$",            re.I),
    re.compile(r"^\s*linea\s+\w{1,5}\s*$",             re.I),
    re.compile(r"^\s*[A-Z\s]{2,40}\s*$"),
    re.compile(r"^\s*corse?\s*:",                       re.I),
    re.compile(r"^\s*fermat[ae]\s*$",                   re.I),
    re.compile(r"^\s*ora\s*$"),
    re.compile(r"^\s*\d{4}\s*$"),
]

_COMMON_LOCATION_WORDS = {
    "del", "della", "di", "da", "al", "all", "alla", "ai", "agli",
    "le", "la", "il", "lo", "e", "per", "via", "viale", "piazza",
    "corso", "strada", "direzione", "direz", "stazione", "autostazione",
    "fermata", "centro", "ospedale", "scalo", "porto",
}


def _is_noise(text: str) -> bool:
    """True se la riga è rumore tipografico o intestazione di pagina."""
    if not text or not text.strip():
        return True
    s = text.strip()
    for pat in _NOISE_PATTERNS:
        if pat.search(s):
            return True
    if s.lower() in _NOISE_KEYWORDS:
        return True
    if _DIGIT_PAT.match(s):
        return True
    return False


def _is_noise_stop(name: str) -> bool:
    """True se il nome della fermata è rumore da scartare."""
    if not name:
        return True
    s = name.strip()
    if len(s) < 3:
        return True
    if re.fullmatch(r"[A-Z]{1,3}", s):
        return True
    low = s.lower()
    for kw in _NOISE_KEYWORDS:
        if kw in low:
            return True
    if re.fullmatch(r"[0-9\s]+", s):
        return True
    if re.search(r"\b(linea|orario|feriale|festivo|valido)\b", low):
        return True
    return False


def _ns(linee: list) -> str:
    return ", ".join(str(l.get("n", "?")) for l in linee)


def _parse_stop(text_part: str, times: list) -> dict:
    return {
        "n": re.sub(r"\s{2,}", " ", text_part).strip(),
        "v": "",
        "o": list(times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF ROUTE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_routes_from_pdf_content(content: str) -> list:
    lines   = content.split("\n")
    fermate = []
    i       = 0

    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()

        if not line:
            i += 1
            continue

        times_in_line = _TIME_PAT.findall(line)
        text_part     = _TIME_PAT.sub("", line)
        text_part     = re.sub(r"[\|\t\*]+",  " ", text_part)
        text_part     = re.sub(r"\s{2,}", " ", text_part).strip().strip("-.,;:")

        if re.search(r"\b(gruppo cadenze|gruppo cadenza|cadenza)\b", text_part, re.I):
            i += 1
            continue

        if re.fullmatch(r"(?:[A-Z]{1,3}\d{1,4}\s+)+[A-Z]{1,3}\d{1,4}", text_part):
            i += 1
            continue

        if times_in_line and text_part and len(text_part) >= 3 and not _is_noise(text_part):
            fermate.append(_parse_stop(text_part, times_in_line))
            i += 1
            continue

        if times_in_line and (not text_part or len(text_part) < 3 or _is_noise(text_part)):
            if fermate:
                last = fermate[-1]
                if not last["o"]:
                    last["o"] = list(times_in_line)
                else:
                    for t in times_in_line:
                        if t not in last["o"]:
                            last["o"].append(t)
            i += 1
            continue

        if not times_in_line and len(text_part) >= 3 and not _is_noise(line):
            lookahead_times: list[str] = []
            j = i + 1
            while j < len(lines) and j < i + 6:
                next_raw   = lines[j].strip()
                if not next_raw:
                    j += 1
                    continue
                next_times = _TIME_PAT.findall(next_raw)
                next_text  = _TIME_PAT.sub("", next_raw)
                next_text  = re.sub(r"[\|\t\*]+", " ", next_text).strip()
                next_text  = re.sub(r"\s{2,}", " ", next_text).strip("-.,;: ")

                if next_times:
                    if not next_text or len(next_text) < 3 or _is_noise(next_text):
                        lookahead_times.extend(next_times)
                        j += 1
                        continue
                    break
                else:
                    if not _is_noise(next_raw):
                        break
                    j += 1

            fermate.append(_parse_stop(text_part, lookahead_times))
            if lookahead_times:
                i = j
            else:
                i += 1
            continue

        i += 1

    deduped    = []
    seen_names: list[str] = []
    for f in fermate:
        n = f["n"].lower().strip()
        if n and n not in seen_names[-3:]:
            deduped.append(f)
            seen_names.append(n)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# LINE NAME + DIRECTION
# ─────────────────────────────────────────────────────────────────────────────
def normalize_line_name(fn: str) -> str:
    s = re.sub(r"\.pdf$", "", fn, flags=re.I).strip()
    s = s.replace("–", "-").replace("—", "-")

    m = re.search(r"\blinea[_\s\-]*([0-9]{1,3}[a-zA-Z]?)\b", s, re.I)
    if m:
        return m.group(1).lstrip("0") or "0"

    m = re.search(
        r"\b(linea[_\s\-]*urbana|urbana[_\s\-]*linea)[_\s\-]*([0-9]{1,3}[a-zA-Z]?)\b",
        s, re.I,
    )
    if m:
        return m.group(2).lstrip("0") or "0"

    m = re.search(r"\b([A-Z][0-9]{1,2})\b", s)
    if m:
        return m.group(1).upper()

    month_names = (
        r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto"
        r"|settembre|ottobre|novembre|dicembre)"
    )
    for m in re.finditer(r"(?<![0-9])([0-9]{1,3}[a-zA-Z]?)(?![0-9])", s):
        candidate = m.group(1)
        prev = s[: m.start()].lower()
        foll = s[m.end() :].lower()
        if re.search(
            r"(?:dal|dall[ae]?|al|del|dei|degli|dalla|dalle)[_\s-]*$", prev
        ) and re.search(rf"^[_\s-]*(?:{month_names})", foll):
            continue
        if re.search(rf"^[_\s-]*(?:{month_names})", foll):
            continue
        return candidate.lstrip("0") or "0"

    m = re.match(r"([0-9]{1,3}[a-zA-Z]?)[_\s\-]", s)
    if m:
        return m.group(1).lstrip("0") or "0"

    m = re.fullmatch(r"([0-9]{1,3}[a-zA-Z]?)", s)
    if m:
        return m.group(1).lstrip("0") or "0"

    m = re.search(r"\b([A-Z]{1,3})\b", s)
    if m:
        return m.group(1)

    clean = re.sub(r"[_\-\s]+", " ", s).strip()
    clean = re.sub(
        r"\b(?:orario|valido|dal|dall|dalla|al|settembre|ottobre|novembre"
        r"|dicembre|gennaio|febbraio|marzo|aprile|maggio|giugno|luglio"
        r"|agosto|settembre)\b.*",
        "",
        clean, flags=re.I,
    ).strip()
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean[:30] if clean else s[:30]


def _detect_direction(filename: str) -> str | None:
    s = filename.lower()
    if re.search(r"(?:^|[_\-.])(ritorno|rit|return)(?:$|[_\-.])", s):
        return "r"
    if re.search(r"(?:^|[_\-.])(andata|and|outward)(?:$|[_\-.])", s):
        return "a"
    return None


def extract_route_description(filename: str, content: str) -> str:
    lines     = [l.strip() for l in content.split("\n") if l.strip()]
    arrow_pat = re.compile(r".{3,}\s*([-–—→/])\s*.{3,}")
    for line in lines[:20]:
        if arrow_pat.search(line) and not _is_noise(line) and len(line) < 120:
            clean = _TIME_PAT.sub("", line).strip()
            if len(clean) > 5:
                if clean.lower().startswith("orario valido"):
                    clean = re.sub(r"(\d{4})([A-ZÀÈÉÌÒÙ])", r"\1 \2", clean)
                    m_year = re.search(r"\d{4}\b(.*)", clean)
                    if m_year:
                        rest = m_year.group(1).strip()
                        if rest:
                            return rest[:120]
                    clean = re.sub(r"(?i)orario\s+valido.*", "", clean).strip()
                return clean[:120]

    for line in lines[:15]:
        if not _is_noise(line) and len(line) > 8 and any(c.isalpha() for c in line):
            clean = _TIME_PAT.sub("", line).strip()
            if len(clean) > 5:
                return clean[:120]

    return f"Linea {filename.replace('.pdf', '')}"


def _split_pdf_line_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    pattern  = re.compile(r"(?im)^LINEA\s+([0-9]{1,3}[A-Za-z]?)\b")
    matches  = list(pattern.finditer(content))
    if not matches:
        return sections
    for idx, m in enumerate(matches):
        start = m.start()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        sections.append((m.group(1), content[start:end]))
    return sections


def _detect_direction_from_section(text: str) -> str | None:
    if re.search(r"\bAndata\b", text, re.I):
        return "a"
    if re.search(r"\bRitorno\b", text, re.I):
        return "r"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS GENERAZIONE JSON
# ─────────────────────────────────────────────────────────────────────────────
def _merge_stop_fragments(fermate: list) -> list:
    merged: list = []
    i_f = 0
    while i_f < len(fermate):
        cur       = fermate[i_f]
        cur_name  = (cur.get("n") or "").strip()
        cur_times = cur.get("o", []) if isinstance(cur.get("o", []), list) else []
        if not cur_times and i_f + 1 < len(fermate):
            nxt       = fermate[i_f + 1]
            nxt_times = nxt.get("o", []) if isinstance(nxt.get("o", []), list) else []
            if nxt_times and cur_name and len(cur_name) < 60:
                merged_name = (cur_name + " " + (nxt.get("n") or "")).strip()
                merged.append({
                    "n": re.sub(r"\s{2,}", " ", merged_name),
                    "v": cur.get("v") or nxt.get("v") or "",
                    "o": nxt_times,
                })
                i_f += 2
                continue
        merged.append({
            "n": re.sub(r"\s{2,}", " ", cur_name),
            "v": cur.get("v", ""),
            "o": cur_times,
        })
        i_f += 1
    return merged


def _clean_fermate(fermate: list, global_stop_names: set) -> list:
    cleaned: list = []
    for f in fermate:
        name = (f.get("n") or "").strip()
        if _is_noise_stop(name):
            continue
        if global_stop_names:
            candidates = difflib.get_close_matches(
                name, list(global_stop_names), n=1, cutoff=0.86
            )
            if candidates:
                f["n"] = candidates[0]
                cleaned.append(f)
                continue
        cleaned.append(f)
        global_stop_names.add(f["n"])
    return cleaned


def _append_section(entry: dict, section_key: str, route_desc: str, fermate: list) -> None:
    existing   = entry.get(section_key, {}) or {}
    existing_f = (
        existing.get("f", []) if isinstance(existing.get("f", []), list) else []
    )
    if not existing_f:
        entry[section_key] = {"p": route_desc, "f": list(fermate)}
    else:
        existing_f.extend(fermate)
        entry[section_key] = {
            "p": existing.get("p", route_desc) or route_desc,
            "f": existing_f,
        }


def _guess_direction(route_desc: str) -> str:
    if re.search(r"ritorno|rit|return", route_desc, re.I):
        return "r"
    if re.search(r"andata|and|outward", route_desc, re.I):
        return "a"
    return "a"


# ─────────────────────────────────────────────────────────────────────────────
# GENERAZIONE JSON
# ─────────────────────────────────────────────────────────────────────────────
def generate_transport_json(pdf_folder_path: str) -> str:
    try:
        json_path = os.path.join(ROOT_PROJECT_DIR, "data", "trasporti.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        force_regen = os.environ.get("FORCE_REGEN", "0").lower() in ("1", "true", "yes")
        if os.path.exists(json_path) and not force_regen:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data     = json.load(f)
                    existing = data.get("linee", [])
                    pdf_count = sum(
                        1
                        for fn in os.listdir(pdf_folder_path)
                        if fn.lower().endswith(".pdf")
                    )
                    if len(existing) >= pdf_count:
                        print("[OK] trasporti.json valido. Riuso.")
                        return json_path
            except Exception as exc:
                print(f"[WARN] trasporti.json corrotto: {exc}. Rigenerazione…")
        elif os.path.exists(json_path) and force_regen:
            print("[INFO] FORCE_REGEN=1 → rigenero trasporti.json")
        else:
            print("[INFO] trasporti.json non trovato. Rigenerazione…")

        linee_dict:        dict[tuple, dict] = {}
        global_stop_names: set[str]          = set()
        categories = ["urbani", "extraurbani", "treni"]

        for category in categories:
            cat_path = os.path.join(pdf_folder_path, category)
            if not os.path.exists(cat_path):
                print(f"[WARN] Cartella mancante: {cat_path}")
                continue

            for file_name in sorted(
                fn for fn in os.listdir(cat_path) if fn.lower().endswith(".pdf")
            ):
                try:
                    full_path = os.path.join(cat_path, file_name)
                    with open(full_path, "rb") as fh:
                        pdf_bytes = fh.read()

                    content        = pdf_content(pdf_bytes)
                    raw_id         = file_name.replace(".pdf", "").strip()
                    file_direction = _detect_direction(file_name)
                    sections       = _split_pdf_line_sections(content)

                    if sections:
                        for section_line, section_text in sections:
                            line_number = normalize_line_name(section_line)
                            direction   = (
                                _detect_direction_from_section(section_text) or file_direction
                            )
                            route_desc  = extract_route_description(file_name, section_text)
                            fermate     = extract_routes_from_pdf_content(section_text)
                            fermate     = _merge_stop_fragments(fermate)
                            fermate     = _clean_fermate(fermate, global_stop_names)

                            key = (category, line_number)
                            if key not in linee_dict:
                                linee_dict[key] = {
                                    "n":            line_number,
                                    "display_name": f"Linea {line_number}",
                                    "categoria":    category,
                                    "a":            {"p": "", "f": []},
                                    "r":            {"p": "", "f": []},
                                    "source":       file_name,
                                }

                            entry = linee_dict[key]
                            if direction in ("a", "r"):
                                _append_section(entry, direction, route_desc, fermate)
                            else:
                                guess = _guess_direction(route_desc)
                                if not entry[guess].get("f"):
                                    _append_section(entry, guess, route_desc, fermate)
                                elif not entry["a"].get("f"):
                                    _append_section(entry, "a", route_desc, fermate)
                                elif not entry["r"].get("f"):
                                    _append_section(entry, "r", route_desc, fermate)
                                else:
                                    _append_section(entry, "a", route_desc, fermate)
                    else:
                        line_number = normalize_line_name(raw_id)
                        direction   = file_direction
                        route_desc  = extract_route_description(file_name, content)
                        fermate     = extract_routes_from_pdf_content(content)
                        fermate     = _merge_stop_fragments(fermate)
                        fermate     = _clean_fermate(fermate, global_stop_names)

                        key = (category, line_number)
                        if key not in linee_dict:
                            linee_dict[key] = {
                                "n":            line_number,
                                "display_name": f"Linea {line_number}",
                                "categoria":    category,
                                "a":            {"p": "", "f": []},
                                "r":            {"p": "", "f": []},
                                "source":       file_name,
                            }

                        entry = linee_dict[key]
                        if direction in ("a", "r"):
                            _append_section(entry, direction, route_desc, fermate)
                        else:
                            guess = _guess_direction(route_desc)
                            if not entry[guess].get("f"):
                                _append_section(entry, guess, route_desc, fermate)
                            elif not entry["a"].get("f"):
                                _append_section(entry, "a", route_desc, fermate)
                            elif not entry["r"].get("f"):
                                _append_section(entry, "r", route_desc, fermate)
                            else:
                                _append_section(entry, "a", route_desc, fermate)

                except Exception as exc:
                    print(f"[ERROR] {file_name}: {exc}")
                    continue

        def _sort_key(linea: dict) -> tuple:
            cat_order = {"urbani": 0, "extraurbani": 1, "treni": 2}
            cat = cat_order.get(linea.get("categoria", ""), 3)
            n   = linea.get("n", "")
            try:
                return (cat, int(re.sub(r"[^0-9]", "", n) or "9999"), n)
            except Exception:
                return (cat, 9999, n)

        linee_list = list(linee_dict.values())
        for linea in linee_list:
            if isinstance(linea.get("a", {}).get("f", []), list):
                linea["a"]["f"] = _dedupe_fermate(linea["a"]["f"])
            if isinstance(linea.get("r", {}).get("f", []), list):
                linea["r"]["f"] = _dedupe_fermate(linea["r"]["f"])

        linee_list.sort(key=_sort_key)

        urbane      = [l for l in linee_list if l.get("categoria") == "urbani"]
        extraurbane = [l for l in linee_list if l.get("categoria") == "extraurbani"]
        treni       = [l for l in linee_list if l.get("categoria") == "treni"]

        transport_data = {
            "servizio":          "Servizio TPL",
            "orario_dal":        str(date.today()),
            "linee":             linee_list,
            "linee_urbane":      urbane,
            "linee_extraurbane": extraurbane,
            "linee_treni":       treni,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transport_data, f, indent=4, ensure_ascii=False)

        print(f"\n[OK] trasporti.json salvato: {len(linee_list)} linee.")
        return json_path

    except Exception as exc:
        import traceback; traceback.print_exc()
        raise RuntimeError(f"Errore generazione trasporti.json: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# DATA ACCESS
# ─────────────────────────────────────────────────────────────────────────────
def get_time_aware_context() -> str:
    context_text = "General Info."
    try:
        now_local = datetime.now().strftime("%Y-%m-%d %H:%M")
        json_path = os.path.join(ROOT_PROJECT_DIR, "data", "trasporti.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        linee_numeri = [l.get("n", "") for l in data.get("linee", []) if l.get("n")]
        if linee_numeri:
            context_text += (
                f"\n[CONTESTO]: Servizio '{data.get('servizio', 'TPL')}' "
                f"valido dal {data.get('orario_dal', 'N/D')}. "
                f"Linee disponibili: {', '.join(linee_numeri[:20])}. "
                f"Orario locale corrente: {now_local}."
            )
    except Exception as exc:
        context_text += f"\n[AVVISO]: Impossibile leggere trasporti.json ({exc})"
    return context_text


def get_transport_data() -> dict:
    try:
        json_path = os.path.join(ROOT_PROJECT_DIR, "data", "trasporti.json")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[ERROR] trasporti.json non trovato!")
        raise HTTPException(status_code=500, detail="File trasporti.json mancante")
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[ERROR] Impossibile caricare trasporti.json: {exc}")
        raise HTTPException(status_code=500, detail=f"Errore caricamento dati: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# STOP MATCHING
# ─────────────────────────────────────────────────────────────────────────────
def _strip_accents(s: str) -> str:
    """Rimuove diacritici: 'Gemòna' → 'Gemona', 'Täufers' → 'Taufers'."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_stop_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    key = name.lower().strip()
    key = re.sub(r"[\(\)\[\],;:\"\'\\/]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    key = _strip_accents(key)
    return key


def _matches_stop_name(query: str, stop_name: str) -> bool:
    q       = _normalize_stop_key(query)
    raw_n   = stop_name or ""
    n_clean = re.sub(r"\(.*?direzion[ei].*?\)", "", raw_n, flags=re.I)
    n_clean = re.sub(r"\bdirezion[ei][:\-\s]*[A-Za-z0-9\s\.\'\-]+", "", n_clean, flags=re.I)
    n       = _normalize_stop_key(n_clean)
    if not q or not n:
        return False

    # Corrispondenza esatta o sottostringa
    if q == n or q in n or n in q:
        return True

    q_tokens = [t for t in q.split() if t and t not in _COMMON_LOCATION_WORDS]
    n_tokens = [t for t in n.split() if t and t not in _COMMON_LOCATION_WORDS]
    if not q_tokens or not n_tokens:
        return False

    # Overlap token esatto
    overlap = set(q_tokens) & set(n_tokens)
    if overlap:
        if len(overlap) >= min(len(q_tokens), len(n_tokens)) / 2:
            return True
        if len(overlap) == 1 and len(q_tokens) == 1:
            if re.search(r"direzion", raw_n or "", re.I):
                return False
            return True

    # Fuzzy fallback: stringa intera (es. "gemona" vs "gemona del friuli")
    if difflib.SequenceMatcher(None, q, n).ratio() >= 0.82:
        return True

    # Fuzzy fallback: token per token (cattura errori tipografici e varianti)
    for qt in q_tokens:
        if len(qt) < 4:
            continue
        for nt in n_tokens:
            if len(nt) < 4:
                continue
            if difflib.SequenceMatcher(None, qt, nt).ratio() >= 0.85:
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# ITINERARY PARSING
# ─────────────────────────────────────────────────────────────────────────────
def _parse_itinerary_query(query: str) -> tuple[str, str]:
    q = query.lower().strip()
    q = re.sub(r"[\?\!\.]+$", "", q)
    patterns = [
        (r"come\s+(?:posso\s+)?arrivare\s+a\s+(.+?)\s+(?:partendo\s+da|da)\s+(.+)", "reverse"),
        (r"come\s+(?:posso\s+)?andare\s+a\s+(.+?)\s+(?:partendo\s+da|da)\s+(.+)",  "reverse"),
        (r"voglio\s+andare\s+a\s+(.+?)\s+(?:partendo\s+da|da)\s+(.+)",             "reverse"),
        (r"arrivare\s+a\s+(.+?)\s+(?:partendo\s+da|da)\s+(.+)",                    "reverse"),
        (r"(?:per\s+arrivare\s+a|per\s+andare\s+a)\s+(.+?)\s+(?:da|partendo\s+da)\s+(.+)", "reverse"),
        (r"(?:partendo\s+da|da)\s+(.+?)\s+(?:arrivare\s+a|a)\s+(.+)",              "normal"),
        (r"^da\s+(.+?)\s+a\s+(.+)$",                                               "normal"),
        (r"^(.+?)\s+da\s+(.+?)\s+$",                                               "reverse"),
    ]
    for pat, order in patterns:
        m = re.search(pat, q)
        if m and m.lastindex >= 2:
            first  = m.group(1).strip().rstrip(" ?.!")
            second = m.group(2).strip().rstrip(" ?.!")
            if order == "reverse":
                return second, first
            return first, second

    if " da " in q and " a " in q:
        m = re.search(r"da\s+(.+?)\s+a\s+(.+)", q)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.search(r"a\s+(.+?)\s+da\s+(.+)", q)
        if m:
            return m.group(2).strip(), m.group(1).strip()
    return "", ""


def _raw_times(fermata: dict) -> list[str]:
    """Restituisce la lista grezza di orari della fermata (ordine originale = ordine corse)."""
    out = []
    for t in (fermata.get("o") or []):
        if not isinstance(t, str):
            continue
        m = _TIME_PAT.search(t)
        if m:
            out.append(m.group(1))
    return out


def _next_corsa(
    fermate: list,
    origin_idx: int,
    dest_idx: int,
    now_min: int | None = None,
) -> tuple[str | None, str | None]:
    """
    Trova la prossima corsa disponibile dall'origine alla destinazione.

    Restituisce (orario_partenza, orario_arrivo).
    L'allineamento è per indice: la N-esima partenza all'origine
    corrisponde al N-esimo orario alla destinazione nella stessa corsa.
    Se non c'è più nessuna corsa oggi, restituisce la prima di domani
    con il suffisso ' (domani)'.
    """
    if now_min is None:
        now     = datetime.now()
        now_min = now.hour * 60 + now.minute

    times_o = _raw_times(fermate[origin_idx])
    times_d = _raw_times(fermate[dest_idx])

    if not times_o:
        return None, None

    first_future: tuple[str | None, str | None] = (None, None)
    first_ever:   tuple[str | None, str | None] = (None, None)

    for i, t_dep in enumerate(times_o):
        dep_min = _time_to_minutes(t_dep)
        if dep_min is None:
            continue
        t_arr = times_d[i] if i < len(times_d) else None

        if first_ever == (None, None):
            first_ever = (t_dep, t_arr)

        if dep_min >= now_min:
            first_future = (t_dep, t_arr)
            break

    if first_future != (None, None):
        return first_future
    # Nessuna corsa rimasta oggi → restituisce la prima di domani
    dep, arr = first_ever
    if dep:
        dep = dep + " (domani)"
    if arr:
        arr = arr + " (domani)"
    return dep, arr


def _find_stop_index_in_fermate(fermate: list, stop_name: str) -> int | None:
    if not fermate:
        return None
    for i, f in enumerate(fermate):
        if _matches_stop_name(stop_name, str(f.get("n", "")).strip()):
            return i
    return None


def _stop_exists_in_data(stop_name: str, transport_data: dict) -> bool:
    """
    Controlla se una fermata (o un nome simile) esiste EFFETTIVAMENTE
    nel JSON dei trasporti.  Restituisce True solo se trovata.
    """
    if not stop_name or not isinstance(transport_data, dict):
        return False
    name_l = stop_name.lower().strip()
    for linea in transport_data.get("linee", []):
        for dir_key in ("a", "r"):
            fermate = ((linea.get(dir_key) or {}).get("f") or [])
            if not isinstance(fermate, list):
                continue
            for f in fermate:
                if _matches_stop_name(name_l, str(f.get("n", "")).strip()):
                    return True
    return False


def _resolve_stop_name(query: str, transport_data: dict) -> str | None:
    """
    Cerca nel JSON la fermata che meglio corrisponde a 'query' e ne restituisce
    il nome ESATTO come appare nel JSON (es. 'cividale' → 'Cividale del Friuli').

    Priorità:
      1. Corrispondenza esatta (case-insensitive, accenti normalizzati)
      2. Sottostringa: query contenuta nel nome della fermata
      3. Sottostringa inversa: nome fermata contenuto nella query
      4. Miglior punteggio difflib tra tutti i token significativi

    Restituisce None se nessuna fermata raggiunge la soglia minima di similarità.
    """
    if not query or not isinstance(transport_data, dict):
        return None

    q_norm = _normalize_stop_key(query)
    if not q_norm:
        return None

    best_name:  str | None = None
    best_score: float      = 0.0

    for linea in transport_data.get("linee", []):
        for dir_key in ("a", "r"):
            fermate = ((linea.get(dir_key) or {}).get("f") or [])
            if not isinstance(fermate, list):
                continue
            for f in fermate:
                raw = str(f.get("n", "")).strip()
                if not raw:
                    continue
                n_norm = _normalize_stop_key(raw)

                # 1. Esatto
                if q_norm == n_norm:
                    return raw

                # 2. Sottostringa diretta
                if q_norm in n_norm:
                    score = len(q_norm) / max(len(n_norm), 1) + 0.5
                    if score > best_score:
                        best_score = score
                        best_name  = raw
                    continue

                # 3. Sottostringa inversa
                if n_norm in q_norm:
                    score = len(n_norm) / max(len(q_norm), 1) + 0.3
                    if score > best_score:
                        best_score = score
                        best_name  = raw
                    continue

                # 4. difflib token vs token
                q_tokens = [t for t in q_norm.split() if t not in _COMMON_LOCATION_WORDS and len(t) >= 3]
                n_tokens = [t for t in n_norm.split() if t not in _COMMON_LOCATION_WORDS and len(t) >= 3]
                if q_tokens and n_tokens:
                    token_scores = [
                        difflib.SequenceMatcher(None, qt, nt).ratio()
                        for qt in q_tokens
                        for nt in n_tokens
                    ]
                    score = max(token_scores) if token_scores else 0.0
                    if score > best_score:
                        best_score = score
                        best_name  = raw

    # Soglia minima: evita falsi positivi su nomi brevissimi
    min_threshold = 0.75 if len(q_norm) >= 5 else 0.90
    if best_score >= min_threshold:
        print(f"[DEBUG] _resolve_stop_name: '{query}' → '{best_name}' (score={best_score:.2f})")
        return best_name

    print(f"[DEBUG] _resolve_stop_name: '{query}' non trovata (best_score={best_score:.2f})")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TIME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _time_to_minutes(t: str) -> int | None:
    try:
        h, m = t.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _find_next_time(times: list, now_min: int | None = None) -> str | None:
    if not times:
        return None
    if now_min is None:
        now     = datetime.now()
        now_min = now.hour * 60 + now.minute
    mins = sorted({
        m for t in times
        for m in [_time_to_minutes(t.strip())]
        if m is not None
    })
    if not mins:
        return None
    for m in mins:
        if m >= now_min:
            return f"{m // 60:02d}:{m % 60:02d}"
    m = mins[0]
    return f"{m // 60:02d}:{m % 60:02d}"


def _normalize_times_list(orari) -> list[str]:
    if not orari:
        return []
    vals: list[str] = []
    for t in orari:
        if not isinstance(t, str):
            continue
        m = _TIME_PAT.search(t)
        if m:
            vals.append(m.group(1))
    seen: set[str]  = set()
    out:  list[str] = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    try:
        out.sort(key=lambda x: (_time_to_minutes(x) or 99999))
    except Exception:
        pass
    return out


def _upcoming_times(orari, now_min: int | None = None, limit: int = 12) -> list[str]:
    if now_min is None:
        now     = datetime.now()
        now_min = now.hour * 60 + now.minute
    norm     = _normalize_times_list(orari)
    upcoming = [t for t in norm if (_time_to_minutes(t) or -1) >= now_min]
    return upcoming[:limit] if upcoming else norm[:limit]


def _pretty_category(cat: str) -> str:
    c       = (cat or "").lower()
    mapping = {"urbani": "Urbani", "extraurbani": "Extraurbani", "treni": "Treni"}
    return mapping.get(c, cat.capitalize() if cat else "Autobus")


def _extract_time_from_question(question: str) -> int | None:
    if not question:
        return None
    m = re.search(r"\b(?:alle|ore)?\s*(\d{1,2}[:.]\d{2}|\d{1,2})\b", question, flags=re.I)
    if not m:
        return None
    t = m.group(1).replace(".", ":")
    try:
        if ":" in t:
            h, mm = t.split(":")
            h_i = int(h) % 24
            m_i = int(mm) % 60
        else:
            h_i = int(t) % 24
            m_i = 0
        return h_i * 60 + m_i
    except Exception:
        return None


def _format_next_departures_for_llm(
    transport_data: dict,
    now_min: int | None = None,
    max_lines: int = 20,
) -> str:
    """Crea un sommario compatto delle prossime partenze per linea."""
    try:
        lines  = transport_data.get("linee", []) if isinstance(transport_data, dict) else []
        parts: list[str] = []
        count  = 0
        for l in lines:
            if count >= max_lines:
                break
            n   = l.get("n", "")
            cat = _pretty_category(l.get("categoria", ""))
            a   = l.get("a", {}) or {}
            r   = l.get("r", {}) or {}
            a_fermate = a.get("f", []) if isinstance(a.get("f", []), list) else []
            r_fermate = r.get("f", []) if isinstance(r.get("f", []), list) else []
            a_times   = _upcoming_times(a_fermate[0].get("o", []), now_min=now_min, limit=3) if a_fermate else []
            r_times   = _upcoming_times(r_fermate[0].get("o", []), now_min=now_min, limit=3) if r_fermate else []
            a_str     = ",".join(a_times) if a_times else "—"
            r_str     = ",".join(r_times) if r_times else "—"
            parts.append(f"Linea {n} [{cat}] - Andata: {a_str} - Ritorno: {r_str}")
            count += 1
        if not parts:
            return ""
        return "PROSSIME_PARTENZE:\n" + "\n".join(parts)
    except Exception as exc:
        print(f"[WARN] format_next_departures_for_llm error: {exc}")
        return ""


def _extract_relevant_transport_snippet(
    question: str,
    transport_data: dict,
    max_items: int = 12,
) -> str:
    """Estrae uno snippet compatto delle linee/fermate più rilevanti rispetto alla domanda."""
    try:
        q     = (question or "").lower()
        linee = transport_data.get("linee", []) if isinstance(transport_data, dict) else []
        snippets: list[str] = []

        # 1) linea esplicita menzionata
        m = re.search(r"\blinea\s*([0-9]{1,3}[A-Za-z]?)\b", q)
        if m:
            target = m.group(1).lstrip("0")
            for l in linee:
                n = str(l.get("n", "")).lstrip("0")
                if n == target:
                    snippets.append(format_transport_data_for_llm({
                        "linee":      [l],
                        "servizio":   transport_data.get("servizio", ""),
                        "orario_dal": transport_data.get("orario_dal", ""),
                    }))
                    break

        # 2) fuzzy search su fermate / descrizioni
        if not snippets:
            tokens = [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", q) if len(t) > 3]
            if tokens:
                scored: list[tuple[int, dict]] = []
                for l in linee:
                    score    = 0
                    a_desc   = ((l.get("a") or {}).get("p", "") or "").lower()
                    r_desc   = ((l.get("r") or {}).get("p", "") or "").lower()
                    combined = f"{a_desc} {r_desc} {str(l.get('n', ''))}"
                    for t in tokens:
                        if t.lower() in combined:
                            score += 1
                    for dk in ("a", "r"):
                        for s in ((l.get(dk) or {}).get("f", []) or []):
                            name = str(s.get("n", "")).lower()
                            for t in tokens:
                                if t.lower() in name:
                                    score += 1
                    if score:
                        scored.append((score, l))
                scored.sort(key=lambda x: x[0], reverse=True)
                for _, l in scored[:max_items]:
                    snippets.append(format_transport_data_for_llm({
                        "linee":      [l],
                        "servizio":   transport_data.get("servizio", ""),
                        "orario_dal": transport_data.get("orario_dal", ""),
                    }))

        # 3) fallback: prime N linee
        if not snippets:
            for l in linee[:max_items]:
                snippets.append(format_transport_data_for_llm({
                    "linee":      [l],
                    "servizio":   transport_data.get("servizio", ""),
                    "orario_dal": transport_data.get("orario_dal", ""),
                }))

        return "\n\n--- RILEVANTI ---\n\n" + "\n\n".join(snippets) if snippets else ""
    except Exception as exc:
        print(f"[WARN] extract_relevant_transport_snippet error: {exc}")
        return ""


def _dedupe_fermate(fermate: list) -> list:
    if not fermate:
        return []
    seen:  dict[str, dict] = {}
    order: list[str]       = []
    for f in fermate:
        name = (f.get("n") or "").strip()
        if not name:
            continue
        key  = re.sub(r"\s+", " ", name).lower()
        norm = _normalize_times_list(f.get("o", []) if isinstance(f.get("o", []), list) else [])
        if key in seen:
            combined       = list(seen[key]["o"]) + list(norm)
            seen[key]["o"] = _normalize_times_list(combined)
        else:
            seen[key] = {"n": name, "v": f.get("v", ""), "o": norm}
            order.append(key)
    return [seen[k] for k in order]


# ─────────────────────────────────────────────────────────────────────────────
# ITINERARY SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def find_direct_itineraries(origin: str, destination: str, data: dict) -> list:
    """Trova linee dirette dove origine e destinazione compaiono in ordine."""
    origin_l = (origin or "").lower().strip()
    dest_l   = (destination or "").lower().strip()
    results  = []
    for linea in data.get("linee", []):
        for dir_key in ("a", "r"):
            dir_data = linea.get(dir_key, {}) if isinstance(linea.get(dir_key, {}), dict) else {}
            fermate  = dir_data.get("f", []) if isinstance(dir_data.get("f", []), list) else []
            oi       = _find_stop_index_in_fermate(fermate, origin_l)
            di       = _find_stop_index_in_fermate(fermate, dest_l)
            if oi is not None and di is not None and oi < di:
                results.append({
                    "n":          linea.get("n"),
                    "categoria":  linea.get("categoria"),
                    "direction":  dir_key,
                    "origin_idx": oi,
                    "dest_idx":   di,
                    "fermate":    fermate,
                })
    return results


def find_one_transfer_itineraries(origin: str, destination: str, data: dict) -> list:
    """Cerca itinerari con un solo trasferimento."""
    origin_l = (origin or "").lower().strip()
    dest_l   = (destination or "").lower().strip()
    linee    = data.get("linee", [])
    results  = []

    origin_lines: list[dict] = []
    dest_lines:   list[dict] = []
    for linea in linee:
        for dir_key in ("a", "r"):
            fermate = ((linea.get(dir_key) or {}).get("f") or [])
            if not isinstance(fermate, list):
                continue
            oi = _find_stop_index_in_fermate(fermate, origin_l)
            di = _find_stop_index_in_fermate(fermate, dest_l)
            if oi is not None:
                origin_lines.append({"line": linea, "dir": dir_key, "idx": oi, "fermate": fermate})
            if di is not None:
                dest_lines.append({"line": linea, "dir": dir_key, "idx": di, "fermate": fermate})

    for ol in origin_lines:
        for dl in dest_lines:
            ol_stops      = ol["fermate"][ol["idx"] + 1:]
            dl_stops      = dl["fermate"][:dl["idx"]]
            ol_stop_names = [str(f.get("n", "")).lower() for f in ol_stops]
            dl_stop_names = [str(f.get("n", "")).lower() for f in dl_stops]
            common        = set(ol_stop_names) & set(dl_stop_names)
            if common:
                transfer_name = next(iter(common))
                transfer_idx_from = next(
                    (i for i, f in enumerate(ol["fermate"])
                     if _matches_stop_name(transfer_name, str(f.get("n", "")))),
                    None,
                )
                transfer_idx_to = next(
                    (i for i, f in enumerate(dl["fermate"])
                     if _matches_stop_name(transfer_name, str(f.get("n", "")))),
                    None,
                )
                if (
                    transfer_idx_from is not None
                    and transfer_idx_from > ol["idx"]
                    and transfer_idx_to is not None
                    and transfer_idx_to < dl["idx"]
                ):
                    results.append({
                        "from_line":         ol["line"],
                        "from_dir":          ol["dir"],
                        "from_idx":          ol["idx"],
                        "to_line":           dl["line"],
                        "to_dir":            dl["dir"],
                        "to_idx":            dl["idx"],
                        "transfer_stop":     transfer_name,
                        "transfer_idx_from": transfer_idx_from,
                        "transfer_idx_to":   transfer_idx_to,
                        "from_segment":      ol["fermate"][ol["idx"]:transfer_idx_from + 1],
                        "to_segment":        dl["fermate"][transfer_idx_to:dl["idx"] + 1],
                    })
    return results


def find_multi_transfer_itineraries(
    origin: str,
    destination: str,
    data: dict,
    max_transfers: int = 3,
) -> list:
    """
    Cerca itinerari con più trasferimenti (fino a max_transfers) usando BFS.
    Restituisce una lista di percorsi; ogni percorso è una lista di segmenti
    con chiavi: line, start_stop, end_stop.
    """
    origin_l = (origin or "").lower().strip()
    dest_l   = (destination or "").lower().strip()
    linee    = data.get("linee", [])

    # Costruisci grafo: stop_name → [(stop_successiva, numero_linea, dir_key)]
    graph: dict[str, list] = {}
    for linea in linee:
        n = linea.get("n", "")
        for dir_key in ("a", "r"):
            fermate = ((linea.get(dir_key) or {}).get("f") or [])
            if not isinstance(fermate, list):
                continue
            for idx in range(len(fermate) - 1):
                f_from = fermate[idx].get("n", "").lower().strip()
                f_to   = fermate[idx + 1].get("n", "").lower().strip()
                if f_from and f_to:
                    graph.setdefault(f_from, []).append((f_to, n, dir_key))

    # BFS
    queue    = deque()
    queue.append((origin_l, [], 0))   # (fermata corrente, path, n_trasferimenti)
    visited  = set()
    results: list = []

    while queue and len(results) < 3:
        current_stop, path, transfers = queue.popleft()
        if transfers > max_transfers:
            continue
        state = (current_stop, transfers)
        if state in visited:
            continue
        visited.add(state)

        for next_stop, line_n, dir_key in graph.get(current_stop, []):
            new_path = path + [{
                "line":       line_n,
                "start_stop": current_stop,
                "end_stop":   next_stop,
            }]
            if _matches_stop_name(dest_l, next_stop):
                results.append(new_path)
                if len(results) >= 3:
                    break
            else:
                # conta trasferimento se si cambia linea
                new_transfers = transfers + (
                    1 if path and path[-1]["line"] != line_n else 0
                )
                if new_transfers <= max_transfers:
                    queue.append((next_stop, new_path, new_transfers))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT TRANSPORT DATA
# ─────────────────────────────────────────────────────────────────────────────
def format_transport_data_for_llm(transport_data: dict) -> str:
    lines = [
        f"SERVIZIO: {transport_data.get('servizio', 'N/A')}",
        f"Valido dal: {transport_data.get('orario_dal', 'N/A')}",
        "",
    ]
    for linea in transport_data.get("linee", []):
        if not isinstance(linea, dict):
            continue
        n   = linea.get("n", "N/A")
        cat = linea.get("categoria", "")
        lines.append(f"━━ LINEA {n} [{cat}] ━━")
        for dir_key, dir_label in [("a", "ANDATA"), ("r", "RITORNO")]:
            dir_data = linea.get(dir_key, {})
            if not isinstance(dir_data, dict):
                dir_data = {}
            percorso = dir_data.get("p", "")
            fermate  = dir_data.get("f", [])
            if not isinstance(fermate, list) or not fermate:
                continue
            lines.append(f"  {dir_label}: {percorso}")
            for fermata in fermate:
                if not isinstance(fermata, dict):
                    continue
                nome  = fermata.get("n", "?")
                orari = fermata.get("o", [])
                if not isinstance(orari, list):
                    orari = []
                o_str = " | ".join(orari[:12]) if orari else "—"
                via   = fermata.get("v", "")
                lines.append(f"    • {nome}{f' ({via})' if via else ''}: {o_str}")
            lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB QUERY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _query_chromadb(question: str, n_results: int = 5) -> str:
    """
    Interroga ChromaDB per trovare i documenti semanticamente più vicini alla domanda.
    Ritorna una stringa formattata con i risultati, o stringa vuota se non disponibile.
    """
    if chromadb is None:
        return ""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            collection = client.get_collection("my_documents")
        except Exception:
            print("[INFO] Collezione 'my_documents' non trovata in ChromaDB.")
            return ""

        results = collection.query(
            query_texts=[question],
            n_results=n_results,
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            return ""

        parts: list[str] = []
        for i, doc in enumerate(results["documents"][0], 1):
            meta   = results["metadatas"][0][i - 1] if results.get("metadatas") else {}
            source = meta.get("source", f"Documento {i}")
            # Limita la lunghezza per non saturare il context window
            snippet = doc[:800].strip()
            parts.append(f"[{i}] {source}:\n{snippet}")
        return "\n\n".join(parts)

    except Exception as exc:
        print(f"[WARN] _query_chromadb: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# GENERAZIONE RISPOSTA – ENTRY POINT PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
def generate_travel_response(
    question: str,
    transport_data: dict,
    context_info: str,
) -> dict:
    """
    Genera la risposta alla domanda dell'utente.

    Ogni domanda viene SEMPRE processata dall'LLM; non esistono risposte
    pre-scritte nel codice. Il modello riceve come contesto:
      1. I dati strutturati estratti da trasporti.json (linee, fermate, orari)
      2. I documenti recuperati semanticamente da ChromaDB
      3. Le circolari scolastiche rilevanti
      4. Il contesto temporale corrente

    Ritorna: {"response": <testo>, "tokens": <stima>}
    """
    global SELECTED_MODEL, LOADED_LLM

    # Sincronizza il modello con la variabile d'ambiente (se cambiata a runtime)
    _env_model = os.environ.get("RAG_LLM_MODEL", "").strip()
    if _env_model and _env_model != SELECTED_MODEL:
        try:
            set_model(_env_model)
        except Exception as e:
            print(f"[WARN] Impossibile impostare il modello da env: {e}")

    print(f"[DEBUG] generate_travel_response: question='{question[:80]}'")

    # ── 0. RISOLUZIONE ITINERARIO VERIFICATA (anti-allucinazione) ────────────
    # Se la domanda è un itinerario (origine → destinazione):
    #   a) Verifica che entrambe le fermate esistano nel JSON.
    #   b) Cerca un collegamento reale con find_direct / find_one_transfer.
    #   c) Se trovato → risposta strutturata SENZA LLM (zero allucinazioni).
    #   d) Se non trovato → risposta negativa diretta SENZA LLM.
    # In nessun caso l'LLM riceve una domanda itinerario con contesto vuoto.
    _origin, _dest = _parse_itinerary_query(question)
    print(f"[DEBUG] _parse_itinerary_query → origin='{_origin}' dest='{_dest}'")

    if _origin and _dest:
        # a) Risolvi i nomi esatti nel JSON (gestisce varianti, accenti, abbreviazioni)
        _origin_resolved = _resolve_stop_name(_origin, transport_data)
        _dest_resolved   = _resolve_stop_name(_dest,   transport_data)
        print(f"[DEBUG] resolved → origin='{_origin_resolved}' dest='{_dest_resolved}'")

        if not _origin_resolved or not _dest_resolved:
            _missing = []
            if not _origin_resolved:
                _missing.append(f'"{_origin}"')
            if not _dest_resolved:
                _missing.append(f'"{_dest}"')
            return {
                "response": (
                    f"Non ho trovato la fermata {' e '.join(_missing)} "
                    "nei dati disponibili. Verifica il nome o contatta "
                    "direttamente l'azienda di trasporto."
                ),
                "tokens": 0,
            }

        # b) Cerca collegamenti reali usando i nomi esatti
        direct    = find_direct_itineraries(_origin_resolved, _dest_resolved, transport_data)
        transfers = find_one_transfer_itineraries(_origin_resolved, _dest_resolved, transport_data) if not direct else []
        print(f"[DEBUG] itinerari diretti={len(direct)} con_trasferimento={len(transfers)}")

        # c) Collegamento trovato → risposta strutturata, zero LLM
        if direct or transfers:
            now     = datetime.now()
            now_min = now.hour * 60 + now.minute
            _lines: list[str] = [
                f"Ecco come andare da {_origin_resolved} a {_dest_resolved}:\n"
            ]
            if direct:
                for it in direct[:3]:
                    fermate   = it["fermate"]
                    oi        = it["origin_idx"]
                    di        = it["dest_idx"]
                    dep, arr  = _next_corsa(fermate, oi, di, now_min)
                    seg       = fermate[oi: di + 1]
                    stops_str = " → ".join(str(f.get("n", "?")) for f in seg)
                    dep_str   = dep if dep else "—"
                    arr_str   = arr if arr else "—"
                    _lines.append(
                        f"🚌 Linea {it['n']} (diretto)\n"
                        f"   Percorso: {stops_str}\n"
                        f"   Prossima partenza da {_origin_resolved}: {dep_str}\n"
                        f"   Arrivo a {_dest_resolved}: {arr_str}"
                    )
            if transfers:
                for it in transfers[:2]:
                    f_line = it["from_line"].get("n", "?")
                    t_line = it["to_line"].get("n", "?")
                    xfer   = it["transfer_stop"].title()
                    dep, xfer_arr = _next_corsa(
                        it["from_line"].get(it["from_dir"], {}).get("f", []),
                        it["from_idx"],
                        it["transfer_idx_from"],
                        now_min,
                    )
                    _, arr = _next_corsa(
                        it["to_line"].get(it["to_dir"], {}).get("f", []),
                        it["transfer_idx_to"],
                        it["to_idx"],
                        now_min,
                    )
                    dep_str      = dep      if dep      else "—"
                    xfer_arr_str = xfer_arr if xfer_arr else "—"
                    arr_str      = arr      if arr      else "—"
                    _lines.append(
                        f"🚌 Linea {f_line} → cambio a {xfer} → Linea {t_line}\n"
                        f"   Prossima partenza da {_origin_resolved}: {dep_str}\n"
                        f"   Arrivo a {xfer}: {xfer_arr_str}\n"
                        f"   Arrivo a {_dest_resolved}: {arr_str}"
                    )
            return {"response": "\n\n".join(_lines), "tokens": 0}

        # d) Nessun collegamento trovato → risposta negativa, zero LLM
        return {
            "response": (
                f"Non ho trovato nessun collegamento diretto né con trasferimento "
                f"tra {_origin_resolved} e {_dest_resolved} nei dati disponibili. "
                "Contatta direttamente l'azienda di trasporto per ulteriori informazioni."
            ),
            "tokens": 0,
        }

    # ── 1. Recupera snippet rilevanti dal JSON dei trasporti ──────────────────
    json_snippet = _extract_relevant_transport_snippet(question, transport_data, max_items=12)
    print(f"[DEBUG] json_snippet: {len(json_snippet)} caratteri")

    # ── 2. Interroga ChromaDB per contesto semantico ──────────────────────────
    chroma_snippet = _query_chromadb(question, n_results=5)
    print(f"[DEBUG] chroma_snippet: {len(chroma_snippet)} caratteri")

    # ── 3. Cerca nelle circolari scolastiche ──────────────────────────────────
    circulars_snippet = search_school_circulars(question, max_results=3)
    print(f"[DEBUG] circulars_snippet: {len(circulars_snippet)} caratteri")

    # ── 4. Assembla il contesto completo ─────────────────────────────────────
    context_parts: list[str] = []
    if json_snippet:
        context_parts.append(
            "=== DATI ORARI E LINEE (trasporti.json) ===\n" + json_snippet
        )
    if chroma_snippet:
        context_parts.append(
            "=== DOCUMENTI CORRELATI (vector DB) ===\n" + chroma_snippet
        )
    if circulars_snippet:
        context_parts.append(
            "=== CIRCOLARI SCOLASTICHE ===\n" + circulars_snippet
        )
    if context_info:
        context_parts.append(
            "=== CONTESTO TEMPORALE ===\n" + context_info
        )

    full_context = (
        "\n\n".join(context_parts)
        if context_parts
        else "(Nessun dato disponibile nei database)"
    )

    # ── 5. Costruisci il system prompt ────────────────────────────────────────
    system_prompt = (
        "Sei un assistente specializzato nel trasporto pubblico locale.\n"
        "Rispondi SEMPRE in italiano.\n\n"
        "REGOLE ASSOLUTE — non derogabili in nessun caso:\n"
        "1. Usa ESCLUSIVAMENTE le fermate, le linee e gli orari presenti nel CONTESTO qui sotto.\n"
        "2. NON inventare MAI fermate, linee, itinerari o orari non presenti nel contesto.\n"
        "3. Se una fermata o una destinazione NON compare nel contesto, di' esplicitamente "
        "   che non hai dati su quella fermata e invita l'utente a contattare l'azienda.\n"
        "4. NON proporre itinerari via treno, automobile o qualsiasi mezzo non documentato "
        "   nel contesto.\n"
        "5. Se il contesto non contiene un percorso diretto o con trasferimento tra le due "
        "   fermate richieste, rispondi: \"Non ho trovato un collegamento tra queste fermate "
        "   nei dati disponibili.\"\n"
        "6. Sii conciso: massimo 10 righe, salvo richiesta esplicita di dettagli completi.\n"
        "7. Quando riporti orari o fermate, elencali in modo ordinato e leggibile.\n\n"
        f"CONTESTO:\n{full_context}"
    )

    # ── 6. Verifica disponibilità LLM ────────────────────────────────────────
    if not SELECTED_MODEL or ChatOllama is None:
        # Nessun LLM configurato: restituisci il contesto grezzo come fallback
        print("[WARN] Nessun LLM disponibile, restituzione contesto grezzo.")
        fallback_text = (
            json_snippet
            or chroma_snippet
            or "Nessun dato trovato per questa domanda. "
               "Imposta RAG_LLM_MODEL per abilitare le risposte intelligenti."
        )
        return {"response": fallback_text, "tokens": 0}

    # ── 7. Chiama l'LLM ───────────────────────────────────────────────────────
    try:
        result = _call_llm(system_prompt, question, temperature=0.0)
        if result and isinstance(result, dict):
            text   = (result.get("text") or "").strip()
            tokens = result.get("tokens", 0)
            if text:
                print(f"[DEBUG] LLM response: {len(text)} caratteri, ~{tokens} token")
                return {"response": text, "tokens": tokens}
    except Exception as exc:
        print(f"[ERROR] LLM generate_travel_response: {exc}")

    # ── 8. Fallback finale ───────────────────────────────────────────────────
    return {
        "response": (
            "Non riesco a elaborare una risposta al momento. "
            "Verifica che il modello LLM sia correttamente configurato e in esecuzione."
        ),
        "tokens": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT PUBBLICO
# ─────────────────────────────────────────────────────────────────────────────
def initialize_system():
    db = init_vector_store(PDF_SOURCE_ROOT)
    print("\n--- SISTEMA PRONTO PER L'INFERENZA ---")
    return db


def generate_answer(question: str) -> str:
    """
    Entry point pubblico per la generazione delle risposte.
    Carica i dati, costruisce il contesto e delega a generate_travel_response.
    Ritorna sempre una stringa.
    """
    try:
        transport_data = get_transport_data()
        context_info   = get_time_aware_context()
        result         = generate_travel_response(question, transport_data, context_info)
        # generate_travel_response ritorna sempre un dict {"response": ..., "tokens": ...}
        if isinstance(result, dict):
            return result.get("response", "")
        # Compatibilità con eventuali chiamate che si aspettano una stringa
        return str(result)
    except Exception as exc:
        print(f"[ERROR] generate_answer: {exc}")
        return "Si è verificato un errore interno. Riprova tra poco."


def unload_model() -> None:
    """Rimuove dalla memoria l'eventuale LLM caricato."""
    global LOADED_LLM, SELECTED_MODEL
    try:
        if LOADED_LLM is not None:
            for attr in ("close", "client"):
                obj = getattr(LOADED_LLM, attr, None)
                if callable(obj):
                    try:
                        obj()
                    except Exception:
                        pass
                elif obj is not None and hasattr(obj, "close"):
                    try:
                        obj.close()
                    except Exception:
                        pass
            LOADED_LLM     = None
            SELECTED_MODEL = "rule-based"
        gc.collect()
        try:
            import torch
            if hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[INFO] unload_model: modello rimosso dalla memoria (se presente).")
    except Exception as exc:
        print(f"[ERROR] unload_model: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# RISPOSTE GENERICHE (domande non relative al trasporto)
# ─────────────────────────────────────────────────────────────────────────────
def _build_generic_system_prompt(context_info: str = "") -> str:
    prompt = (
        "Sei un assistente utile, amichevole e informativo.\n"
        "Rispondi sempre in italiano.\n"
        "Sii breve, chiaro e diretto al punto."
    )
    if context_info:
        prompt += f"\n\nCONTESTI AGGIUNTIVI:\n{context_info}"
    return prompt


def generate_generic_response(question: str, context_info: str = "") -> str:
    """
    Genera una risposta generica a qualsiasi domanda usando il modello selezionato.
    Fallback testuale se nessun LLM è configurato o disponibile.
    """
    try:
        sys_prompt = _build_generic_system_prompt(context_info)
        result     = _call_llm(sys_prompt, question)
        if result and isinstance(result, dict):
            return result.get("text", "")
        elif result and isinstance(result, str):
            return result
        return (
            "Mi dispiace, al momento non riesco a dare una risposta dettagliata. "
            "Prova a riformulare la domanda o fornisci più dettagli."
        )
    except Exception as e:
        print(f"[ERROR] generate_generic_response: {e}")
        return "Mi dispiace, sto avendo difficoltà a rispondere in questo momento. Riprova tra poco."