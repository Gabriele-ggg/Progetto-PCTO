#!/usr/bin/env python3
"""
pdf_orari_to_json.py
--------------------
Legge tutti i PDF di orari bus (formato TPL FVG) da una cartella
tramite LangChain (PyPDFDirectoryLoader) e produce un JSON strutturato:

  {
    "linea": "81",
    "direzione": "Andata",
    "percorso": "Via Colugna - Centro - via Colugna",
    "validita": "dall'11 settembre 2025",
    "tipo_servizio": "URBANO feriale",
    "fermate": [
      {
        "codice": "UD178",
        "nome": "UDINE via Colugna (fronte 147, chiesa B.Vergine di Fatima)",
        "orari": ["08:30", "08:50", "09:10", ...]
      },
      ...
    ]
  }

Dipendenze:
    pip install langchain-community pypdf

Modifica PDF_FOLDER e OUTPUT_JSON, poi esegui:
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

PDF_FOLDER  = r"D:\prova_git\Progetto-PCTO\pdf"
OUTPUT_JSON = r"D:\prova_git\Progetto-PCTO\orari_output.json"

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Pattern ───────────────────────────────────────────────────────────────────

# Intestazione pagina: "LINEA 81 (Andata)"
RE_HEADER = re.compile(
    r"LINEA\s+(\S+)\s*\(([^)]+)\)",   # gruppo 1 = numero, gruppo 2 = direzione
    re.IGNORECASE,
)

# Riga del percorso: "Via Colugna - Centro - via Colugna"  (subito dopo l'header)
RE_PERCORSO = re.compile(
    r"^([A-Za-zÀ-ÿ][^\n]{3,80}?)\s+Orario valido",
    re.IGNORECASE | re.MULTILINE,
)

# Validità: "Orario valido dall'11 settembre 2025"
RE_VALIDITA = re.compile(
    r"[Oo]rario\s+valido\s+([\w\s']+\d{4})",
)

# Tipo servizio: "U500   URBANO feriale"
RE_SERVIZIO = re.compile(
    r"[A-Z]\d{3}\s+(URBANO\s+\w+|EXTRAURBANO\s+\w+)",
    re.IGNORECASE,
)

# Riga fermata: codice (es. UD178) + nome + orari
# Struttura testo estratto da PyPDF:
#   " UD178 UDINE via Colugna (fronte 147, chiesa\nB.Vergine di Fatima)\n08:30 08:50 ..."
# oppure tutto su una riga:
#   " UD178 UDINE via Colugna (fronte 147) 08:30 08:50 09:10"
RE_FERMATA_LINE = re.compile(
    r"^\s*([A-Z0-9]{5})\s+(.+?)(?=\s{2,}|\t|$)((?:\s+\d{2}:\d{2})+)?$",
    re.MULTILINE,
)

RE_TIME = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")


# ─────────────────────────────────────────────────────────────────────────────
# Parser per pagina singola
# ─────────────────────────────────────────────────────────────────────────────

def parse_page(text: str) -> dict | None:
    """
    Analizza il testo di una pagina e restituisce un dict con:
      linea, direzione, percorso, validita, tipo_servizio, fermate
    oppure None se la pagina non contiene dati di orario.
    """
    # Deve contenere "LINEA"
    m_header = RE_HEADER.search(text)
    if not m_header:
        return None

    linea     = m_header.group(1).strip()
    direzione = m_header.group(2).strip()

    # Percorso
    m_percorso = RE_PERCORSO.search(text)
    percorso = m_percorso.group(1).strip() if m_percorso else ""

    # Validità
    m_val = RE_VALIDITA.search(text)
    validita = m_val.group(1).strip() if m_val else ""

    # Tipo servizio
    m_serv = RE_SERVIZIO.search(text)
    tipo_servizio = m_serv.group(1).strip() if m_serv else ""

    # ── Estrazione fermate ────────────────────────────────────────────────────
    # Il testo estratto da PyPDF da questa tipologia di PDF ha questa forma:
    #
    #  UD178 UDINE via Colugna (fronte 147, chiesa \nB.Vergine di Fatima)\n
    #  08:30 08:50 09:10 09:30 ...
    #
    # oppure codice + nome + orari tutti su una riga.
    # Strategia: cerchiamo blocchi che iniziano con un codice fermata (5 car)
    # e raccogliamo tutti gli orari che seguono fino al prossimo codice.

    # Normalizza: unisci righe spezzate del nome fermata
    # (le righe "orfane" senza codice che precedono una riga di orari)
    lines = text.splitlines()

    fermate: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Cerca un codice fermata all'inizio della riga (es. "UD178 UDINE via ...")
        m = re.match(r"^([A-Z0-9]{4,6})\s+(.*)", line)
        if m:
            codice = m.group(1)
            nome_part = m.group(2).strip()

            # Raccogli eventuale continuazione del nome sulla riga successiva
            # (riga che NON inizia con codice e NON contiene solo orari)
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # Se la riga successiva è solo orari o è vuota, stop
                if not next_line:
                    j += 1
                    break
                if re.match(r"^([A-Z0-9]{4,6})\s+", next_line):
                    break  # prossima fermata
                if re.match(r"^(\d{2}:\d{2}\s*)+$", next_line):
                    break  # riga di orari pura
                # È la continuazione del nome
                nome_part += " " + next_line
                j += 1

            i = j  # avanza il cursore

            # Estrai orari dal nome (a volte sono inline dopo il nome)
            orari_inline = RE_TIME.findall(nome_part)
            # Rimuovi gli orari dalla stringa del nome
            nome_pulito = RE_TIME.sub("", nome_part).strip().rstrip(".,;:-")
            # Normalizza spazi multipli nel nome
            nome_pulito = re.sub(r"\s{2,}", " ", nome_pulito)

            fermate.append({
                "codice": codice,
                "nome": nome_pulito,
                "orari": orari_inline,
            })
        else:
            # Riga di orari pura → appartiene all'ultima fermata
            orari = RE_TIME.findall(line)
            if orari and fermate:
                fermate[-1]["orari"].extend(orari)
            i += 1

    # Deduplica orari mantenendo l'ordine
    for f in fermate:
        seen = []
        for o in f["orari"]:
            if o not in seen:
                seen.append(o)
        f["orari"] = seen

    # Filtra fermate senza nome significativo o senza orari
    fermate = [f for f in fermate if f["nome"] and f["orari"]]

    if not fermate:
        return None

    return {
        "linea": linea,
        "direzione": direzione,
        "percorso": percorso,
        "validita": validita,
        "tipo_servizio": tipo_servizio,
        "fermate": fermate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Merge pagine della stessa linea+direzione
# ─────────────────────────────────────────────────────────────────────────────

def merge_direzioni(pages: list[dict]) -> list[dict]:
    """
    Raggruppa le pagine con stessa linea+direzione e fonde gli orari
    delle fermate (le pagine successive contengono altre corse della stessa tabella).
    """
    grouped: dict[str, dict] = {}

    for page in pages:
        key = f"{page['linea']}|{page['direzione']}"
        if key not in grouped:
            grouped[key] = {
                "linea": page["linea"],
                "direzione": page["direzione"],
                "percorso": page["percorso"],
                "validita": page["validita"],
                "tipo_servizio": page["tipo_servizio"],
                "fermate": [],
            }
            # Inizializza fermate
            for f in page["fermate"]:
                grouped[key]["fermate"].append({
                    "codice": f["codice"],
                    "nome": f["nome"],
                    "orari": list(f["orari"]),
                })
        else:
            # Aggiungi orari alle fermate già presenti (match per codice)
            existing = {f["codice"]: f for f in grouped[key]["fermate"]}
            for f in page["fermate"]:
                if f["codice"] in existing:
                    for o in f["orari"]:
                        if o not in existing[f["codice"]]["orari"]:
                            existing[f["codice"]]["orari"].append(o)
                else:
                    grouped[key]["fermate"].append({
                        "codice": f["codice"],
                        "nome": f["nome"],
                        "orari": list(f["orari"]),
                    })

    # Ordina orari di ogni fermata
    result = []
    for entry in grouped.values():
        for f in entry["fermate"]:
            f["orari"] = sorted(set(f["orari"]))
        result.append(entry)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    folder = Path(PDF_FOLDER)
    if not folder.is_dir():
        raise SystemExit(f"Cartella non trovata: {folder.resolve()}")

    log.info("Caricamento PDF da: %s", folder.resolve())
    docs = PyPDFDirectoryLoader(str(folder)).load()

    if not docs:
        raise SystemExit("Nessun PDF trovato o nessun testo estratto.")

    # Raggruppa pagine per file sorgente
    files: dict[str, list] = defaultdict(list)
    for doc in docs:
        name = Path(doc.metadata.get("source", "sconosciuto")).name
        files[name].append(doc.page_content)

    output_documenti = []
    errori = []

    for nome_file, pagine in sorted(files.items()):
        log.info("Elaborazione: %s (%d pagine)", nome_file, len(pagine))
        parsed_pages = []
        for testo in pagine:
            try:
                result = parse_page(testo)
                if result:
                    parsed_pages.append(result)
            except Exception as exc:
                log.warning("Errore su pagina di %s: %s", nome_file, exc)

        if not parsed_pages:
            log.warning("Nessun dato estratto da: %s", nome_file)
            errori.append({"file": nome_file, "errore": "nessun dato estratto"})
            continue

        linee = merge_direzioni(parsed_pages)
        output_documenti.append({
            "file": nome_file,
            "analizzato_il": datetime.now().isoformat(timespec="seconds"),
            "linee": linee,
        })

    output = {
        "generato_il": datetime.now().isoformat(timespec="seconds"),
        "cartella": str(folder.resolve()),
        "pdf_elaborati": len(output_documenti),
        "pdf_con_errori": len(errori),
        "errori": errori,
        "documenti": output_documenti,
    }

    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("JSON salvato in: %s", out_path.resolve())


if __name__ == "__main__":
    main()