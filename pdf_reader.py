#!/usr/bin/env python3
"""
pdf_orari_to_json.py
--------------------
Legge tutti i PDF di una cartella tramite LangChain (PyPDFDirectoryLoader)
e ne estrae orari, tipo di trasporto, fermate, linee e date,
salvando tutto in un file JSON.

Dipendenze:
    pip install langchain-community pypdf

Modifica PDF_FOLDER e OUTPUT_JSON qui sotto, poi esegui:
    python pdf_orari_to_json.py
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from langchain_community.document_loaders import PyPDFDirectoryLoader

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURA QUI
# ─────────────────────────────────────────────────────────────────────────────

PDF_FOLDER  = r"D:\prova_git\Progetto-PCTO\pdf"           # cartella con i PDF
OUTPUT_JSON = r"D:\prova_git\Progetto-PCTO\orari_output.json"

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Pattern
RE_TIME = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
RE_DATE = re.compile(r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b")
RE_LINE = re.compile(r"(?:linea|line|route|percorso|servizio)\s*[:\-]?\s*([A-Z0-9][\w\-/]{0,10})", re.IGNORECASE)
RE_STOP = re.compile(r"(?:fermata|stazione|stop|station|partenza|arrivo|da|a|from|to)\s*[:\-]?\s*([A-ZÀÁÈÉÌÍÒÓÙÚ][a-zàáèéìíòóùú\s\-']{2,40})", re.IGNORECASE)

TRANSPORT_KEYWORDS = {
    "treno":     ["treno", "train", "ferrovia", "railway", "fs", "trenitalia", "italo", "freccia", "intercity", "regionale"],
    "autobus":   ["autobus", "bus", "pullman", "coach", "corriera", "navetta", "shuttle", "cotral", "flixbus"],
    "aereo":     ["aereo", "volo", "flight", "aeroporto", "airport", "airline", "ryanair", "easyjet"],
    "traghetto": ["traghetto", "ferry", "nave", "ship", "motonave", "aliscafo"],
    "metro":     ["metro", "metropolitana", "subway", "underground"],
    "tram":      ["tram", "tramway", "tranvai"],
}


def detect_transport(text: str) -> list[str]:
    tl = text.lower()
    found = [t for t, kws in TRANSPORT_KEYWORDS.items() if any(k in tl for k in kws)]
    return found or ["sconosciuto"]

def extract_times(text: str) -> list[str]:
    return sorted(set(RE_TIME.findall(text)))

def extract_dates(text: str) -> list[str]:
    return sorted(set(RE_DATE.findall(text)))

def extract_lines(text: str) -> list[str]:
    return sorted(set(m.strip() for m in RE_LINE.findall(text)))

def extract_stops(text: str) -> list[str]:
    skip = {"da", "a", "al", "del", "di", "e", "il", "la", "le", "gli", "i"}
    raw = [m.strip().rstrip(".,;:") for m in RE_STOP.findall(text)]
    return sorted({s for s in raw if len(s) > 2 and s.lower() not in skip})


def main():
    folder = Path(PDF_FOLDER)
    if not folder.is_dir():
        raise SystemExit(f"Cartella non trovata: {folder.resolve()}")

    log.info("Caricamento PDF da: %s", folder.resolve())
    docs = PyPDFDirectoryLoader(str(folder)).load()

    if not docs:
        raise SystemExit("Nessun PDF trovato o nessun testo estratto.")

    # Raggruppa pagine per file
    grouped = defaultdict(lambda: {"testo": "", "pagine": [], "metadati": {}})
    for doc in docs:
        name = Path(doc.metadata.get("source", "sconosciuto")).name
        grouped[name]["testo"] += doc.page_content + "\n"
        grouped[name]["pagine"].append(doc.metadata.get("page", "?"))
        grouped[name]["metadati"] = {k: v for k, v in doc.metadata.items() if k != "page"}

    risultati = []
    for nome, dati in sorted(grouped.items()):
        log.info("Analisi: %s", nome)
        testo = dati["testo"]
        risultati.append({
            "file": nome,
            "analizzato_il": datetime.now().isoformat(timespec="seconds"),
            "metadati": dati["metadati"],
            "numero_pagine": len(dati["pagine"]),
            "tipo_trasporto": detect_transport(testo),
            "linee_rilevate": extract_lines(testo),
            "fermate_stazioni": extract_stops(testo),
            "orari_trovati": extract_times(testo),
            "date_trovate": extract_dates(testo),
        })

    output = {
        "generato_il": datetime.now().isoformat(timespec="seconds"),
        "cartella": str(folder.resolve()),
        "pdf_elaborati": len(risultati),
        "documenti": risultati,
    }

    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("JSON salvato in: %s", out_path.resolve())


if __name__ == "__main__":
    main()