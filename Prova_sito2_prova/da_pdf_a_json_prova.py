import json
import os
import re
import requests
from pathlib import Path

# ============================================================
# PARSING PDF (Con LLM locale Ollama)
# ============================================================

def _interroga_ollama_per_json(testo_pagina: str, modello: str = "qwen2.5:1.5b") -> dict:
    """
    Invia il testo estratto dalla pagina del PDF ad Ollama, chiedendo di 
    strutturarlo in formato JSON secondo uno schema rigido.
    """
    prompt_sistema = (
        "Sei un assistente specializzato nell'estrazione di dati strutturati da tabelle orarie di autobus.\n"
        "Analizza il testo fornito ed estrai le fermate e gli orari delle corse presenti.\n"
        "Devi restituire OBBLIGATORIAMENTE ed ESCLUSIVAMENTE un oggetto JSON valido, senza blocchi di codice markdown (no ```json), "
        "senza testo introduttivo o spiegazioni.\n\n"
        "Lo schema JSON deve essere esattamente il seguente:\n"
        "{\n"
        "  \"corse\": [\n"
        "    {\n"
        "      \"id\": \"Corsa_1\",\n"
        "      \"giorni\": [\"Tutti i giorni\"],\n"
        "      \"fermate\": [\n"
        "        {\"orario\": \"08:30\", \"nome\": \"Nome Fermata A\"},\n"
        "        {\"orario\": \"08:55\", \"nome\": \"Nome Fermata B\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Nota bene: Non cambiare i nomi delle chiavi nel JSON, rimani fedele a questo schema."
    )

    payload = {
        "model": modello,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Ecco il testo della pagina del tabellone:\n\n{testo_pagina}"}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,  # Zero allucinazioni, massima precisione deterministica
            "num_ctx": 4096      # Contesto ampio per contenere intere pagine
        },
        "format": "json"  # Forziamo Ollama a rispondere solo con JSON valido
    }

    try:
        res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=90)
        res.raise_for_status()
        risposta_testo = res.json().get("message", {}).get("content", "").strip()
        
        # Pulizia di sicurezza nel caso in cui persistano stringhe di formattazione markdown
        risposta_testo = re.sub(r'^```json\s*|```$', '', risposta_testo, flags=re.MULTILINE).strip()
        
        return json.loads(risposta_testo)
    except Exception as e:
        print(f"    [AVVISO] Errore nell'estrazione LLM per questa pagina: {e}")
        return {"corse": []}


def _estrai_corse_da_pdf(pdf_file: Path, tipologia: str) -> dict:
    """
    Analizza il testo del PDF usando il modello Ollama per estrarre orari e fermate,
    mantenendo l'estrazione euristica automatica per il numero della linea e la direzione dal nome del file.
    """
    from langchain_community.document_loaders.pdf import PyPDFLoader
    
    try:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
    except Exception as e:
        print(f"    [ERRORE] Lettura PDF {pdf_file.name}: {e}")
        return None

    stem = pdf_file.stem
    
    # Estrazione numero linea e direzione dal nome del file (Rimane euristica ed affidabile)
    match_num = re.search(r'\b\d+[A-Za-z]?\b', stem)
    numero_linea = match_num.group(0) if match_num else stem

    direzione_raw = re.sub(r'(?i)\blinea\b', '', stem)
    if match_num:
        direzione_raw = direzione_raw.replace(numero_linea, '', 1)
    
    direzione = re.sub(r'^[\s\-_]+|[\s\-_]+$', '', direzione_raw).replace('_', ' ').strip()
    if not direzione:
        direzione = "Direzione non specificata"

    tutte_le_corse = []
    contatore_corsa = 1

    # Utilizziamo l'LLM locale per elaborare il testo pagina per pagina
    for idx, page in enumerate(pages):
        if not page.page_content.strip():
            continue
            
        print(f"      → Analisi intelligente pagina {idx+1}/{len(pages)} con LLM...")
        dati_strutturati = _interroga_ollama_per_json(page.page_content)
        
        # Uniamo le corse estratte standardizzando gli ID corsa in sequenza progressiva
        for corsa in dati_strutturati.get("corse", []):
            if corsa.get("fermate"):
                corsa["id"] = f"Corsa_{contatore_corsa}"
                tutte_le_corse.append(corsa)
                contatore_corsa += 1

    return {
        "nome": str(numero_linea),
        "direzione": direzione,
        "tipo": tipologia,
        "corse": tutte_le_corse
    }


def converti_struttura_pdf_in_json(cartella_root_pdf: str, percorso_output_json: str):
    """
    Scansiona la cartella principale (es. 'pdf') e cerca al suo interno le sottocartelle
    'urbani' ed 'extraurbani'. Estrae i dati da tutti i PDF trovati e genera un unico JSON.
    """
    root_path = Path(cartella_root_pdf)
    
    if not root_path.exists() or not root_path.is_dir():
        print(f"[ERRORE] La cartella principale '{cartella_root_pdf}' non esiste nella stessa directory dello script.")
        print(f"Crea una cartella chiamata '{cartella_root_pdf}' con al suo interno 'urbani' e 'extraurbani'.")
        return

    # Definizione delle sottocartelle da cercare e la loro relativa tipologia da assegnare nel JSON
    sottocartelle_target = [
        {"nome": "urbani", "tipo": "urbani"},
        {"nome": "extraurbani", "tipo": "extraurbani"}
    ]

    struttura_finale = {"linee": []}
    file_totali_da_elaborare = []

    # 1. Raccolta preliminare di tutti i file per mostrare un log accurato
    for sub in sottocartelle_target:
        sub_path = root_path / sub["nome"]
        if sub_path.exists() and sub_path.is_dir():
            pdf_trovati = sorted([f for f in sub_path.iterdir() if f.is_file() and f.suffix.lower() == '.pdf'])
            for pdf in pdf_trovati:
                file_totali_da_elaborare.append({"file_path": pdf, "tipo": sub["tipo"], "sub_folder": sub["nome"]})
        else:
            print(f"[AVVISO] Sottocartella attesa non trovata o non valida: {root_path.name}/{sub['nome']}")

    if not file_totali_da_elaborare:
        print(f"[AVVISO] Nessun file PDF trovato nelle sottocartelle 'urbani' o 'extraurbani' di '{cartella_root_pdf}'.")
        return

    totale_file = len(file_totali_da_elaborare)
    print(f"[*] Trovati {totale_file} file PDF totali da elaborare nelle sottocartelle.")

    # 2. Elaborazione ciclica di tutti i file trovati
    for idx, info in enumerate(file_totali_da_elaborare, 1):
        file_pdf = info["file_path"]
        tipo_linea = info["tipo"]
        nome_cartella = info["sub_folder"]
        
        print(f"\n[{idx}/{totale_file}] Avvio estrazione [{tipo_linea.upper()}] da: {cartella_root_pdf}/{nome_cartella}/{file_pdf.name}")
        
        risultato_linea = _estrai_corse_da_pdf(file_pdf, tipo_linea)
        
        if risultato_linea and risultato_linea.get("corse"):
            struttura_finale["linee"].append(risultato_linea)
            print(f"[✓] Dati estratti con successo da: {file_pdf.name} (Trovate {len(risultato_linea['corse'])} corse)")
        else:
            print(f"[AVVISO] Nessun dato valido estratto da: {file_pdf.name}")

    # 3. Scrittura del file JSON finale cumulativo
    if struttura_finale["linee"]:
        try:
            with open(percorso_output_json, 'w', encoding='utf-8') as f:
                json.dump(struttura_finale, f, indent=2, ensure_ascii=False)
            print(f"\n[✓] PROCESSO COMPLETATO: Generato file globale unico in: {percorso_output_json}")
        except Exception as e:
            print(f"\n[ERRORE] Impossibile scrivere il file JSON complessivo: {e}")
    else:
        print("\n[ERRORE] Nessun dato valido raccolto, file JSON globale non generato.")


# ============================================================
# AVVIO DELLO SCRIPT
# ============================================================
if __name__ == "__main__":
    # Cartella principale di partenza
    cartella_padre = "pdf" 
    json_output_globale = "trasporti_generato_globale.json"
    
    # Assicurati che Ollama sia attivo nel terminale prima di lanciare lo script:
    # ollama run qwen2.5:1.5b
    
    converti_struttura_pdf_in_json(
        cartella_root_pdf=cartella_padre,
        percorso_output_json=json_output_globale
    )