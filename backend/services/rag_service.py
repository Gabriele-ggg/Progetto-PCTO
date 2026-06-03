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

try:
    DB_INSTANCE = initialize_system()
except Exception as e:
    print("ERRORE ALL'AVVIO DEL BACKEND:", repr(e))
    DB_INSTANCE = None


# Provider Anthropic rimosso.


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
# Esempi:
#   RAG_LLM_MODEL=mistral:7b                          → Ollama locale
#   RAG_LLM_MODEL=rule-based                          → nessun LLM (default)
SELECTED_MODEL: str = AI_MODEL_TO_USE

# Riferimento globale a un'istanza ChatOllama caricata (solo per provider Ollama)
LOADED_LLM = None


def get_selected_model() -> str:
    try:
        return SELECTED_MODEL
    except Exception:
        return "rule-based"


def _init_model_from_env() -> None:
    """Inizializza LOADED_LLM se RAG_LLM_MODEL punta a un modello Ollama."""
    global LOADED_LLM, SELECTED_MODEL
    m = SELECTED_MODEL
    if m == "rule-based":
        return  # niente da caricare
    if ChatOllama is None:
        print(f"[WARN] RAG_LLM_MODEL={m} ma ChatOllama non disponibile.")
        return
    try:
        LOADED_LLM = ChatOllama(model=m, temperature=0.2, top_p=0.8)
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
    model_name = (model_name or "rule-based").strip()
    print(f"[DEBUG] set_model called: '{model_name}'")

    AI_MODEL_TO_USE = model_name

    if model_name == "rule-based":
        SELECTED_MODEL = "rule-based"
        LOADED_LLM = None
        print("[INFO] Modello impostato: rule-based")
        return

    # Ollama locale
    if ChatOllama is None:
        print("[WARN] ChatOllama non disponibile; mantengo rule-based")
        SELECTED_MODEL = "rule-based"
        LOADED_LLM = None
        return
    try:
        LOADED_LLM     = ChatOllama(model=model_name, temperature=0.2, top_p=0.8)
        SELECTED_MODEL = model_name
        print(f"[INFO] Modello impostato: {model_name} (provider Ollama)")
    except Exception as exc:
        print(f"[ERROR] Impossibile caricare il modello Ollama '{model_name}': {exc}")
        # Imposta comunque SELECTED_MODEL così il frontend sa che è stato selezionato
        # anche se al momento non disponibile
        SELECTED_MODEL = model_name
        print(f"[INFO] Modello '{model_name}' selezionato ma non disponibile al momento")


def _call_llm(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.2,
) -> dict | None:
    """
    Invia un messaggio al modello LLM selezionato e restituisce la risposta con metadati.

    Ritorna un dizionario: {"text": risposta, "tokens": numero_token_approssimato}
    o None se nessun LLM è configurato.
    """
    global SELECTED_MODEL, LOADED_LLM
    print(f"[DEBUG] _call_llm: SELECTED_MODEL='{SELECTED_MODEL}'")

    if SELECTED_MODEL == "rule-based":
        return None

    # ── Provider Ollama ───────────────────────────────────────────────────────
    if ChatOllama is None:
        print("[WARN] ChatOllama non disponibile.")
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as exc:
        print(f"[WARN] LangChain non disponibile: {exc}")
        return None

    llm = LOADED_LLM
    if llm is None:
        # Prova a istanziarlo al volo
        try:
            llm = ChatOllama(model=SELECTED_MODEL, temperature=temperature, top_p=0.8)
            LOADED_LLM = llm
        except Exception as exc:
            print(f"[WARN] Impossibile creare istanza Ollama per '{SELECTED_MODEL}': {exc}")
            return None

    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        resp = llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        
        # Stima approssimativa dei token: media tra parole e caratteri/4
        words = len(text.split())
        chars = len(text)
        tokens_estimate = max(words, max(1, chars // 4))
        
        return {"text": text, "tokens": tokens_estimate}
    except Exception as exc:
        print(f"[WARN] _call_llm Ollama error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ROBOTS.TXT – COMMENTATO (le funzioni seguenti sono disabilitate)
# ─────────────────────────────────────────────────────────────────────────────
# Eccezioni robots.txt:
#   - RobotsNotFound: quando robot.txt non è raggiungibile
#   - DisallowedByRobots: quando robot.txt vieta l'accesso
#
# Funzioni disabilitate:
#   - _get_robots_parser(): recupera e analizza robots.txt
#   - _check_robots(): verifica che lo scraping sia consentito
#   - fetch_with_robots(): fetcha URL rispettando robots.txt
#
# Motivo: la sezione robots.txt è stata commentata per semplificare l'architettura.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ESTRAZIONE TESTO DA HTML
# ─────────────────────────────────────────────────────────────────────────────
_BLOCK_TAGS = {
    "script", "style", "noscript", "nav", "footer",
    "header", "aside", "iframe", "svg", "form",
}


def _extract_text_from_html(html: str) -> str:
    """
    Estrae testo significativo dall'HTML.

    Usa BeautifulSoup se disponibile; altrimenti rimuove i tag con regex.
    """
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

    # Fallback regex
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FETCH CON ROBOTS.TXT – COMMENTATO
# ─────────────────────────────────────────────────────────────────────────────
# Funzione disabilitata: fetch_with_robots(url, allowed_hosts, user_agent, session)
# Recupera un URL rispettando robots.txt.
# La richiesta viene eseguita SOLO se:
#   1. Il sito pubblica un robots.txt raggiungibile (HTTP 200).
#   2. Il robots.txt permette l'accesso all'URL per l'user_agent.
#
# Returns: requests.Response
# Raises: RobotsNotFound, DisallowedByRobots, requests.HTTPError
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPE & SAVE – COMMENTATO (dipende da robots.txt e fetch_with_robots)
# ─────────────────────────────────────────────────────────────────────────────
# Funzione disabilitata: scrape_and_save(target_urls, allowed_hosts, user_agent, json_path, session)
# Effettua lo scraping dei target_urls rispettando robots.txt e salva i risultati in JSON.
#
# Regole:
#   - Se allowed_hosts è None e SCRAPE_ALLOWED_HOSTS è vuoto,
#     gli host dei target_urls vengono considerati automaticamente autorizzati.
#   - Lo scraping di un URL viene saltato se il sito non espone robots.txt
#     (stato 'no_robots') oppure se robots.txt lo vieta (stato 'disallowed').
#
# Returns: (json_path, results) dove results è un dict url -> entry.
# ─────────────────────────────────────────────────────────────────────────────


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
        
        # Le circolari saranno caricate on-demand alla prima ricerca
        
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
# SCHOOL CIRCULARS LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Cache per le circolari scolastiche caricate
SCHOOL_CIRCULARS_CACHE = {}
CIRCULARS_LOADED = False

def load_school_circulars_simple():
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
        
        pdf_files = [f for f in os.listdir(circulars_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"[INFO] Nessun PDF trovato nella cartella circolari: {circulars_folder}")
            CIRCULARS_LOADED = True
            return
        
        print(f"[INFO] Caricamento {len(pdf_files)} circolari scolastiche...")
        
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(circulars_folder, pdf_file)
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                content = pdf_content(pdf_bytes)
                
                if not content.strip():
                    print(f"[WARN] Contenuto vuoto dalla circolare: {pdf_file}")
                    continue
                
                # Memorizza il contenuto della circolare nel cache
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
# SCHOOL CIRCULARS SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def search_school_circulars(query: str, max_results: int = 3) -> str:
    """
    Cerca le circolari scolastiche rilevanti per la query.
    Ritorna un testo formattato con i risultati o una stringa vuota se nessun risultato.
    Carica le circolari on-demand alla prima ricerca.
    """
    try:
        # Carica le circolari se non sono state caricate
        load_school_circulars_simple()
        
        if not SCHOOL_CIRCULARS_CACHE:
            return ""
        
        # Ricerca semplice per keyword nelle circolari
        query_lower = query.lower()
        results = []
        
        for filename, content in SCHOOL_CIRCULARS_CACHE.items():
            # Dividi il contenuto in paragrafi
            paragraphs = content.split('\n\n')
            
            # Cerca paragrafi che contengono parole dalla query
            matching_paragraphs = []
            for para in paragraphs:
                para_lower = para.lower()
                # Controlla se il paragrafo contiene almeno una parola significativa dalla query
                if any(word in para_lower for word in query_lower.split() if len(word) > 2):
                    matching_paragraphs.append(para.strip())
            
            if matching_paragraphs:
                # Aggiungi i migliori risultati
                for para in matching_paragraphs[:2]:  # Max 2 paragrafi per circolare
                    results.append((filename, para[:500]))  # Max 500 caratteri
        
        if not results:
            return ""
        
        # Formatta i risultati
        formatted = "CIRCOLARI SCOLASTICHE RILEVANTI:\n"
        for i, (filename, text) in enumerate(results[:max_results], 1):
            formatted += f"\n[{i}] Da: {filename}\n{text}...\n"
        
        return formatted
    
    except Exception as e:
        print(f"[ERROR] Errore ricerca circolari: {e}")
        return ""
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
    re.compile(r"^\s*pagina\s*\d+\s*$",              re.I),
    re.compile(r"^\s*pag\.?\s*\d+\s*(/\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*[-_=*]{3,}\s*$"),
    re.compile(r"^\s*\[pagina\]\s*$",                 re.I),
    re.compile(r"^\s*(andata|ritorno)\s*$",           re.I),
    re.compile(r"^\s*linea\s+\w{1,5}\s*$",            re.I),
    re.compile(r"^\s*[A-Z\s]{2,40}\s*$"),
    re.compile(r"^\s*corse?\s*:",                      re.I),
    re.compile(r"^\s*fermat[ae]\s*$",                  re.I),
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
    """Restituisce una stringa con i numeri di linea separati da virgola."""
    return ", ".join(str(l.get("n", "?")) for l in linee)


def _parse_stop(text_part: str, times: list) -> dict:
    return {
        "n": re.sub(r"\s{2,}", " ", text_part).strip(),
        "v": "",
        "o": list(times),
    }


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
# HELPERS GENERAZIONE JSON (estratti per evitare duplicazione)
# ─────────────────────────────────────────────────────────────────────────────
def _merge_stop_fragments(fermate: list) -> list:
    """
    Unisce coppie (nome senza orari, orari senza nome) prodotte da split
    tipografici del PDF.
    """
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
    """Rimuove fermate rumore e applica correzione fuzzy dei nomi."""
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
    """Aggiunge (o estende) la sezione andata/ritorno di una linea."""
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

        linee_dict:       dict[tuple, dict] = {}
        global_stop_names: set[str]         = set()
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
def get_time_aware_context(db) -> str:
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
def _normalize_stop_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    key = name.lower().strip()
    key = re.sub(r"[\(\)\[\],;:\"\'\\/]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _matches_stop_name(query: str, stop_name: str) -> bool:
    q     = _normalize_stop_key(query)
    raw_n = stop_name or ""
    n_clean = re.sub(r"\(.*?direzion[ei].*?\)", "", raw_n, flags=re.I)
    n_clean = re.sub(r"\bdirezion[ei][:\-\s]*[A-Za-z0-9\s\.\'\-]+", "", n_clean, flags=re.I)
    n = _normalize_stop_key(n_clean)
    if not q or not n:
        return False
    if q == n or q in n or n in q:
        return True

    q_tokens = [t for t in q.split() if t and t not in _COMMON_LOCATION_WORDS]
    n_tokens = [t for t in n.split() if t and t not in _COMMON_LOCATION_WORDS]
    if not q_tokens or not n_tokens:
        return False

    overlap = set(q_tokens) & set(n_tokens)
    if overlap:
        if len(overlap) >= min(len(q_tokens), len(n_tokens)) / 2:
            return True
        if len(overlap) == 1 and len(q_tokens) == 1:
            if re.search(r"direzion", raw_n or "", re.I):
                return False
            return True
    return False


def _find_stop_index_in_fermate(fermate: list, stop_name: str) -> int | None:
    if not fermate:
        return None
    for i, f in enumerate(fermate):
        if _matches_stop_name(stop_name, str(f.get("n", "")).strip()):
            return i
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
    seen: set[str] = set()
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
    c = (cat or "").lower()
    mapping = {"urbani": "Urbani", "extraurbani": "Extraurbani", "treni": "Treni"}
    return mapping.get(c, cat.capitalize() if cat else "Autobus")


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
            combined        = list(seen[key]["o"]) + list(norm)
            seen[key]["o"]  = _normalize_times_list(combined)
        else:
            seen[key] = {"n": name, "v": f.get("v", ""), "o": norm}
            order.append(key)
    return [seen[k] for k in order]


# ─────────────────────────────────────────────────────────────────────────────
# SCHOOL CIRCULARS SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def search_school_circulars(query: str, max_results: int = 3) -> str:
    """
    Cerca le circolari scolastiche rilevanti per la query.
    Ritorna un testo formattato con i risultati o una stringa vuota se nessun risultato.
    Carica le circolari on-demand alla prima ricerca.
    """
    try:
        if chromadb is None:
            return ""
        
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            collection = client.get_collection("school_circulars")
        except Exception:
            # Collezione non esiste ancora - carichiamo le circolari on-demand
            print("[INFO] Collezione circolari non trovata, caricamento on-demand...")
            load_school_circulars(client)
            try:
                collection = client.get_collection("school_circulars")
            except Exception:
                # Nessuna circolare disponibile
                return ""
        
        # Query semantica nella collezione delle circolari
        results = collection.query(
            query_texts=[query],
            n_results=max_results
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return ""
        
        # Formatta i risultati
        formatted = "CIRCOLARI SCOLASTICHE RILEVANTI:\n"
        
        for i, doc in enumerate(results['documents'][0], 1):
            metadata = results['metadatas'][0][i-1] if results.get('metadatas') else {}
            source = metadata.get('source', 'Sconosciuta')
            formatted += f"\n[{i}] Da: {source}\n{doc[:500]}...\n"
        
        return formatted
    
    except Exception as e:
        print(f"[ERROR] Errore ricerca circolari: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ITINERARY SEARCH
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
                lines.append(f"    \u2022 {nome}{f' ({via})' if via else ''}: {o_str}")
            lines.append("")
    return "\n".join(lines)


def generate_travel_response(
    question: str,
    transport_data: dict,
    context_info: str,
) -> dict:
    """
    Passa la domanda dell'utente all'LLM selezionato fornendo come contesto i dati di trasporto.
    Ritorna un dizionario: {"response": testo, "tokens": numero_token}
    Se nessun LLM è configurato (rule-based) restituisce un messaggio di fallback.
    """
    global SELECTED_MODEL, LOADED_LLM

    # Rilegge l'env live nel caso in cui set_model() non sia ancora stato chiamato
    if SELECTED_MODEL == "rule-based":
        _env_model = os.environ.get("RAG_LLM_MODEL", "").strip()
        if _env_model and _env_model != "rule-based":
            print(f"[INFO] generate_travel_response: RAG_LLM_MODEL='{_env_model}' trovato nell'env live → set_model()")
            set_model(_env_model)

    print(f"[DEBUG] generate_travel_response: SELECTED_MODEL='{SELECTED_MODEL}' question='{question[:60]}'")

    # Nessun LLM configurato: fallback testuale
    if SELECTED_MODEL == "rule-based":
        fallback_msg = (
            "Nessun modello LLM configurato. "
            "Avvia il sistema con RAG_LLM_MODEL=<nome_modello> oppure "
            "seleziona un modello dalla sidebar."
        )
        return {"response": fallback_msg, "tokens": len(fallback_msg.split())}

    # Costruisce il contesto dei dati di trasporto da passare all'LLM
    transport_ctx = format_transport_data_for_llm(transport_data)
    
    # Ricerca le circolari scolastiche rilevanti
    circulars_ctx = search_school_circulars(question)
    
    # Costruisce il system prompt con dati di trasporto e circolari
    sys_prompt = (
        "Sei un assistente di viaggio per il TPL (Trasporto Pubblico Locale) e assistente scolastico.\n"
        "LINGUA OBBLIGATORIA: Rispondi SEMPRE e SOLO in italiano.\n"
        "Usa i dati nella sezione DATI TRASPORTI per rispondere a domande su linee, orari e fermate.\n"
        "Se disponibili, usa anche le CIRCOLARI SCOLASTICHE per rispondere a domande su regolamenti o circolari.\n"
        "Per domande non riguardanti i trasporti rispondi comunque in modo utile e amichevole.\n\n"
        "DATI TRASPORTI:\n"
        + transport_ctx
    )
    
    # Aggiungi le circolari se trovate
    if circulars_ctx.strip():
        sys_prompt += "\n\n" + circulars_ctx

    try:
        llm_result = _call_llm(sys_prompt, question)
        if llm_result and isinstance(llm_result, dict):
            return llm_result
        elif llm_result and isinstance(llm_result, str):
            # Compatibilità con vecchio formato
            return {"response": llm_result, "tokens": len(llm_result.split())}
        
        fallback = (
            "Mi dispiace, il modello non ha prodotto una risposta. "
            "Prova a riformulare la domanda."
        )
        return {"response": fallback, "tokens": len(fallback.split())}
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[ERROR] generate_travel_response: {exc}")
        raise HTTPException(status_code=500, detail=f"Errore generazione risposta: {exc}") from exc



def initialize_system():
    db = init_vector_store(PDF_SOURCE_ROOT)
    print("\n--- SISTEMA PRONTO PER L'INFERENZA ---")
    return db


def generate_answer(question: str) -> str:
    try:
        transport_data = get_transport_data()
        context_info   = get_time_aware_context(transport_data)
        return generate_travel_response(question, transport_data, context_info)
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


def _build_generic_system_prompt(context_info: str = "") -> str:
    """
    Costruisce il system prompt per risposte generiche.
    """
    prompt = (
        "Sei un assistente utile, amichevole e informativo.\n"
        "Rispondi sempre in italiano.\n"
        "Sii breve, chiaro e diritto al punto."
    )
    if context_info:
        prompt += f"\n\nCONTESTI AGGIUNTIVI:\n{context_info}"
    return prompt


def generate_generic_response(question: str, context_info: str = "") -> str:
    """
    Genera una risposta generica a qualsiasi domanda usando il modello selezionato.

    Usa _call_llm (Ollama o Anthropic in base a SELECTED_MODEL).
    Fallback testuale se nessun LLM è configurato o disponibile.
    """
    try:
        sys_prompt = _build_generic_system_prompt(context_info)
        result = _call_llm(sys_prompt, question)
        
        # Gestisci il nuovo formato dict di _call_llm
        if result and isinstance(result, dict):
            return result.get("text", "")
        elif result and isinstance(result, str):
            # Compatibilità con vecchio formato
            return result
        
        return (
            "Mi dispiace, al momento non riesco a dare una risposta dettagliata. "
            "Prova a riformulare la domanda o fornisci più dettagli."
        )
    except Exception as e:
        print(f"[ERROR] generate_generic_response: {e}")
        return "Mi dispiace, sto avendo difficoltà a rispondere in questo momento. Riprova tra poco."
