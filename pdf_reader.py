#!/usr/bin/env python3
"""
pdf_orari_to_json.py
--------------------
Legge tutti i PDF di una cartella e ne estrae informazioni su orari,
tipo di trasporto e altri dati rilevanti, salvando il risultato in un
file JSON strutturato.

Uso:
    python pdf_orari_to_json.py <cartella_pdf> [output.json]

Dipendenze:
    pip install pdfplumber pypdf

Esempio:
    python pdf_orari_to_json.py ./orari output_orari.json
"""

import sys
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime

# ── Librerie PDF ──────────────────────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    sys.exit("Installa pdfplumber:  pip install pdfplumber")

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("Installa pypdf:  pip install pypdf")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern di riconoscimento
# ─────────────────────────────────────────────────────────────────────────────

# Orari nel formato HH:MM  (es. 08:30, 23:59)
PATTERN_TIME = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")

# Date comuni italiane / internazionali
PATTERN_DATE = re.compile(
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b"   # 12/05/2025, 5-3-25
    r"|\b(\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b"          # 2025-05-12
)

# Linee / numeri di servizio
PATTERN_LINE = re.compile(
    r"(?:linea|line|route|percorso|servizio)\s*[:\-]?\s*([A-Z0-9][\w\-/]{0,10})",
    re.IGNORECASE,
)

# Fermate / stazioni
PATTERN_STOP = re.compile(
    r"(?:fermata|stazione|stop|station|partenza|arrivo|da|a|from|to)\s*[:\-]?\s*"
    r"([A-ZÀÁÈÉÌÍÒÓÙÚ][a-zàáèéìíòóùú\s\-']{2,40})",
    re.IGNORECASE,
)

# Tipo di trasporto (parole chiave)
TRANSPORT_KEYWORDS: dict[str, list[str]] = {
    "treno":    ["treno", "train", "ferrovia", "railway", "rail", "fs", "trenitalia",
                 "italo", "freccia", "intercity", "regionale", "r ", "ic ", "ec "],
    "autobus":  ["autobus", "bus", "pullman", "coach", "corriera", "navetta",
                 "shuttle", "tram", "metro", "atm", "atv", "ctm", "cotral",
                 "flixbus", "flixcoach"],
    "aereo":    ["aereo", "volo", "flight", "aeroporto", "airport", "compagnia aerea",
                 "airline", "alitalia", "ryanair", "easyjet", "ita"],
    "traghetto":["traghetto", "ferry", "nave", "ship", "motonave", "aliscafo",
                 "catamarano"],
    "metro":    ["metro", "metropolitana", "subway", "underground", "u-bahn"],
    "tram":     ["tram", "tramway", "tranvai"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Funzioni di estrazione
# ─────────────────────────────────────────────────────────────────────────────

def detect_transport_type(text: str) -> list[str]:
    """Rileva il/i tipo/i di trasporto presenti nel testo."""
    text_low = text.lower()
    found = []
    for transport, keywords in TRANSPORT_KEYWORDS.items():
        if any(kw in text_low for kw in keywords):
            found.append(transport)
    return found if found else ["sconosciuto"]


def extract_times(text: str) -> list[str]:
    """Estrae tutti gli orari HH:MM univoci dal testo."""
    times = sorted(set(PATTERN_TIME.findall(text)))
    # Ricostruisce la stringa completa HH:MM
    full_times = sorted(set(re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", text)))
    return full_times


def extract_dates(text: str) -> list[str]:
    """Estrae le date trovate nel testo."""
    raw = re.findall(
        r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b",
        text,
    )
    return sorted(set(raw))


def extract_lines(text: str) -> list[str]:
    """Estrae i numeri/codici di linea."""
    return sorted(set(m.strip() for m in PATTERN_LINE.findall(text)))


def extract_stops(text: str) -> list[str]:
    """Estrae nomi di fermate/stazioni."""
    raw = [m.strip().rstrip(".,;:") for m in PATTERN_STOP.findall(text)]
    # Filtra stringhe troppo brevi o generiche
    cleaned = [s for s in raw if len(s) > 2 and s.lower() not in
               {"da", "a", "al", "del", "di", "e", "il", "la", "le", "gli", "i"}]
    return sorted(set(cleaned))


def extract_tables(path: Path) -> list[list[list[str | None]]]:
    """
    Usa pdfplumber per estrarre tabelle strutturate dal PDF.
    Restituisce una lista di tabelle (ogni tabella = lista di righe).
    """
    tables: list = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if table:
                        tables.append(table)
    except Exception as exc:
        log.warning("Impossibile estrarre tabelle da %s: %s", path.name, exc)
    return tables


def extract_metadata(path: Path) -> dict:
    """Legge i metadati del PDF tramite pypdf."""
    meta: dict = {}
    try:
        reader = PdfReader(str(path))
        info = reader.metadata
        if info:
            for key in ("/Title", "/Author", "/Subject", "/Creator",
                        "/Producer", "/CreationDate", "/ModDate"):
                val = info.get(key)
                if val:
                    meta[key.lstrip("/")] = str(val)
        meta["pages"] = len(reader.pages)
    except Exception as exc:
        log.warning("Impossibile leggere metadati di %s: %s", path.name, exc)
    return meta


def full_text_from_pdf(path: Path) -> str:
    """Estrae tutto il testo del PDF con pdfplumber."""
    parts: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
    except Exception as exc:
        log.warning("Errore lettura testo di %s: %s", path.name, exc)
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Funzione principale di analisi per singolo PDF
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pdf(path: Path) -> dict:
    """Analizza un singolo PDF e restituisce un dizionario con i dati estratti."""
    log.info("Elaborazione: %s", path.name)

    text = full_text_from_pdf(path)
    metadata = extract_metadata(path)
    tables = extract_tables(path)

    orari = extract_times(text)
    date = extract_dates(text)
    transport_types = detect_transport_type(text)
    lines = extract_lines(text)
    stops = extract_stops(text)

    # Tabelle semplificate (prima 20 righe per non appesantire il JSON)
    tables_summary = []
    for i, table in enumerate(tables):
        tables_summary.append({
            "tabella_index": i + 1,
            "righe_totali": len(table),
            "anteprima": table[:20],  # max 20 righe
        })

    return {
        "file": path.name,
        "percorso": str(path.resolve()),
        "analizzato_il": datetime.now().isoformat(timespec="seconds"),
        "metadati": metadata,
        "tipo_trasporto": transport_types,
        "linee_rilevate": lines,
        "fermate_stazioni": stops,
        "orari_trovati": orari,
        "date_trovate": date,
        "tabelle": tables_summary,
        "testo_completo_caratteri": len(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"Errore: '{folder}' non è una cartella valida.")

    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else folder / "orari_output.json"

    pdf_files = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    if not pdf_files:
        sys.exit(f"Nessun file PDF trovato in '{folder}'.")

    log.info("Trovati %d PDF in '%s'", len(pdf_files), folder)

    results = []
    errori = []

    for pdf_path in pdf_files:
        try:
            data = analyze_pdf(pdf_path)
            results.append(data)
        except Exception as exc:
            log.error("Errore su %s: %s", pdf_path.name, exc)
            errori.append({"file": pdf_path.name, "errore": str(exc)})

    output = {
        "generato_il": datetime.now().isoformat(timespec="seconds"),
        "cartella_sorgente": str(folder.resolve()),
        "pdf_elaborati": len(results),
        "pdf_con_errori": len(errori),
        "errori": errori,
        "documenti": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("JSON salvato in: %s", output_path)
    log.info("Elaborati %d PDF, %d errori.", len(results), len(errori))


if __name__ == "__main__":
    main()