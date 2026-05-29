"""
pipeline_completa.py — Edizione Itinerari Personalizzati (Generazione JSON e Vector DB)
- Aggiunta generazione del file JSON con struttura identica al file di riferimento
- Caricamento del JSON e generazione immediata dei Vector DB Embedding (ChromaDB)
- Mantenuti gli endpoint /api/reload, /api/status e il sistema di Routing RAG potenziato
"""

import argparse
import json
import time
import sys
import os
import re
from pathlib import Path

# ============================================================
# STEP 0 — Generazione File JSON (Stessa struttura dell'allegato)
# ============================================================

def step0_genera_json_strutturato(path_output: str = "trasporti_generato.json"):
    """
    Genera un file JSON con la stessa esatta struttura del file 
    'trasporti_non_copiare_prendere_spunto.json'.
    Questo è il Formato A previsto dal parser.
    """
    _titolo("STEP 0 — Generazione JSON Strutturato")
    
    # Costruisci qui la struttura dati esatta (puoi inserire un ciclo 
    # se leggi i dati da un DB esterno, CSV o PDF)
    struttura_json = {
        "servizio": "Servizio Urbano UDINE",
        "orario_dal": "2025-09-11",
        "linee": [
            {
                "n": "1",
                "a": {
                    "p": "Via Chiusaforte-Ospedale-Chiavris-1°Maggio-FS-Gervasutta",
                    "f": [
                        {
                            "c": "UD159",
                            "n": "UDINE via Chiusaforte (parcheggio scambiatore)",
                            "ci": "UDINE",
                            "v": "via Chiusaforte (parcheggio scambiatore)",
                            "o": [
                                "06:10",
                                "06:25",
                                "06:36",
                                "06:54",
                                "07:05"
                                # Aggiungi gli altri orari reali qui...
                            ]
                        },
                        {
                            "c": "UD188",
                            "n": "UDINE via Colugna (fronte civico 63)",
                            "ci": "UDINE",
                            "v": "via Colugna (fronte civico 63)",
                            "o": [
                                "06:10",
                                "06:25",
                                "06:36",
                                "06:54",
                                "07:05"
                            ]
                        }
                    ]
                }
                # Se presente, qui puoi aggiungere anche la chiave "r" per il Ritorno:
                # "r": { "p": "...", "f": [ ... ] }
            }
        ]
    }

    try:
        with open(path_output, "w", encoding="utf-8") as f:
            json.dump(struttura_json, f, ensure_ascii=False, indent=4)
        print(f"  [✓] File JSON generato con successo e salvato in: {path_output}")
        return path_output
    except Exception as e:
        print(f"  [ERRORE] Impossibile generare il file JSON: {e}")
        return None


# ============================================================
# STEP 1 — Test connessione Ollama
# ============================================================

def step1_test_ollama(model: str = "qwen3.5:9b") -> bool:
    import requests
    _titolo("STEP 1 — Test Ollama + LLM locale")
    try:
        requests.get("http://localhost:11434", timeout=3)
        print("  [OK] Ollama raggiungibile")
        return True
    except requests.ConnectionError:
        print("  [ERRORE] Ollama non raggiungibile. Avvialo con: ollama serve")
        return False

# ============================================================
# STEP 2 — Ingestione e chunking da JSON pronto
# ============================================================

def step2_ingestione(path: str = "trasporti_non_copiare_prendere_spunto.json") -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _titolo("STEP 2 — Ingestione + Chunking")
    t_start = time.time()

    print(f"  Caricamento file JSON pre-esistente: {path}")
    documenti = _carica_json(path)

    if not documenti:
        print("  [ATTENZIONE] Nessun documento caricato. Controlla il file JSON.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documenti)
    print(f"  [✓] {len(chunks)} chunk generati in {time.time() - t_start:.2f}s")
    return chunks


def _carica_json(path: str):
    """
    Supporta i vari formati, in primis il FORMATO A (struttura "linee" con campo "n", "a", "r")
    """
    from langchain_core.documents import Document

    try:
        with open(path, encoding="utf-8") as f:
            dati = json.load(f)
    except Exception as e:
        print(f"    [ERRORE] Lettura JSON in {path}: {e}")
        return []

    documenti = []
    linee = dati.get("linee")

    if linee:
        primo = linee[0] if linee else {}

        if "n" in primo:
            # ── FORMATO A ── campo "n" = nome linea, "a" = andata, "r" = ritorno
            for linea in linee:
                nome_linea = str(linea.get("n", "N/D"))
                tipo_linea = linea.get("tipo", "autobus")

                for dir_key, dir_label in [("a", "Andata"), ("r", "Ritorno")]:
                    direzione_obj = linea.get(dir_key)
                    if not direzione_obj:
                        continue

                    percorso = direzione_obj.get("p", "")
                    fermate  = direzione_obj.get("f", [])

                    if not fermate:
                        continue

                    n_corse = max(len(f.get("o", [])) for f in fermate)

                    for i in range(n_corse):
                        fermate_testo_parts = []
                        for fermata in fermate:
                            orari = fermata.get("o", [])
                            orario = orari[i] if i < len(orari) else ""
                            if orario:  
                                nome_f = fermata.get("n", fermata.get("c", "N/D"))
                                fermate_testo_parts.append(f"{orario} — {nome_f}")

                        if not fermate_testo_parts:
                            continue

                        corsa_id = f"Linea_{nome_linea}_{dir_label}_Corsa_{i + 1}"
                        testo = (
                            f"Linea: {nome_linea} ({tipo_linea})\n"
                            f"Direzione: {dir_label}\n"
                            f"Percorso: {percorso}\n"
                            f"Corsa: {corsa_id}\n"
                            f"Fermate:\n" + "\n".join(fermate_testo_parts)
                        )
                        documenti.append(Document(
                            page_content=testo,
                            metadata={
                                "linea": nome_linea,
                                "direzione": dir_label,
                                "corsa": corsa_id,
                                "tipologia": tipo_linea,
                            }
                        ))
        else:
            # ── FORMATO B ──
            corse_iter = [
                (linea.get("nome", "N/D"), linea.get("tipo", "N/D"),
                 linea.get("direzione", "N/D"), corsa)
                for linea in linee
                for corsa in linea.get("corse", [])
            ]
            documenti.extend(_doc_da_corse(corse_iter))

    else:
        # ── FORMATO C ──
        corse_raw = dati.get("corse", [])
        if not corse_raw:
            print("    [ERRORE] Il JSON non contiene né 'linee' né 'corse'.")
            return []

        corse_iter = []
        for corsa in corse_raw:
            id_corsa = corsa.get("id", "")
            m = re.match(r"Linea_(\w+)_(Andata|Ritorno|Navetta|[^_]+)_Corsa_\d+", id_corsa)
            nome_linea = m.group(1) if m else id_corsa
            direzione  = m.group(2) if m else "N/D"
            corse_iter.append((nome_linea, "autobus", direzione, corsa))

        documenti.extend(_doc_da_corse(corse_iter))

    print(f"    [✓] {len(documenti)} documenti caricati da {path}")
    return documenti


def _doc_da_corse(corse_iter) -> list:
    from langchain_core.documents import Document

    documenti = []
    for nome_linea, tipo, direzione, corsa in corse_iter:
        fermate_testo = "\n".join(
            f"{f['orario']} — {f['nome']}"
            for f in corsa.get("fermate", [])
        )
        testo = (
            f"Linea: {nome_linea} ({tipo})\n"
            f"Direzione: {direzione}\n"
            f"Corsa: {corsa.get('id', 'N/D')} | Giorni: {', '.join(corsa.get('giorni', []))}\n"
            f"Fermate:\n{fermate_testo}"
        )
        documenti.append(Document(
            page_content=testo,
            metadata={
                "linea": nome_linea,
                "direzione": direzione,
                "corsa": corsa.get("id", "N/D"),
                "tipologia": tipo,
            }
        ))
    return documenti

# ============================================================
# STEP 3 — Vector DB (ChromaDB Embeddings)
# ============================================================

CHROMA_BASE = "./chroma_db"

def step3_vettoriale(chunks: list):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_chroma import Chroma

    _titolo("STEP 3 — Vector DB (ChromaDB)")

    if not chunks:
        print("  [ATTENZIONE] Nessun chunk da indicizzare. Step 3 saltato.")
        return

    persist_dir = f"{CHROMA_BASE}_nomic"
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="docs_nomic",
        )
        _stato_pipeline["vectorstore"] = vectorstore
        print(f"  [✓] Database vettoriale pronto con nomic-embed-text:latest ({len(chunks)} chunk).")
        print(f"  [✓] Percorso: {persist_dir}")
    except Exception as e:
        print(f"  [ERRORE CRITICO] Creazione database: {e}")

# ============================================================
# STEP 4 — Server FastAPI & Endpoints
# ============================================================

_stato_pipeline: dict = {
    "chunks": [],
    "vectorstore": None,
    "llm_model": "qwen3.5:9b",
    "json_path": "trasporti_generato.json",
}

def _crea_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    import requests as _requests

    app = FastAPI()
    static_dir = Path(__file__).resolve().parent
    index_file = static_dir / "index.html"

    @app.get("/")
    def serve_index():
        if index_file.exists():
            return FileResponse(index_file, media_type="text/html")
        return {"error": "index.html non trovato"}

    @app.get("/api/status")
    def frontend_status():
        chunks = _stato_pipeline.get("chunks", [])
        linee_set = {}
        tipi_count = {}
        corse_set = set()

        for chunk in chunks:
            meta = chunk.metadata
            linea     = meta.get("linea", "")
            tipologia = meta.get("tipologia", "")
            corsa_id  = meta.get("corsa", "")

            if linea:
                linee_set[linea] = tipologia
            if tipologia:
                tipi_count[tipologia] = tipi_count.get(tipologia, 0) + 1
            if corsa_id:
                corse_set.add(corsa_id)

        linee_list = [{"linea": l, "tipo": t} for l, t in linee_set.items()]
        linee_list.sort(key=lambda x: x["linea"])

        return {
            "totale_corse": len(corse_set),
            "totale_linee": len(linee_set),
            "pdf_analizzati": [],
            "generato_il": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tipi": tipi_count,
            "linee": linee_list,
            "db_pronto": _stato_pipeline.get("vectorstore") is not None,
            "embedding": "nomic-embed-text:latest",
            "llm": _stato_pipeline.get("llm_model"),
        }

    @app.post("/api/reload")
    def frontend_reload():
        try:
            path_json = _stato_pipeline.get("json_path", "trasporti_generato.json")
            # Prima rigenereiamo il JSON strutturato
            step0_genera_json_strutturato(path_output=path_json)
            # Poi eseguiamo l'ingestione
            chunks = step2_ingestione(path=path_json)
            _stato_pipeline["chunks"] = chunks
            step3_vettoriale(chunks)
            db_ok = _stato_pipeline.get("vectorstore") is not None
            return {
                "status": "success" if db_ok else "warning",
                "chunks": len(chunks),
                "db_pronto": db_ok,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat")
    async def frontend_chat(data: dict):
        message = data.get("message", "")

        context = ""
        avviso_contesto = ""
        try:
            vs = _stato_pipeline.get("vectorstore")
            if vs:
                res = vs.similarity_search_with_score(message, k=6)
                if res:
                    for doc, _ in res:
                        context += f"{doc.page_content}\n---\n"
                else:
                    avviso_contesto = "[Nessun dato rilevante trovato nel database per questa richiesta.]\n\n"
            else:
                avviso_contesto = "[Database vettoriale non disponibile. Ricarica i dati tramite /api/reload.]\n\n"
        except Exception as e:
            avviso_contesto = f"[Errore nel recupero del contesto: {e}]\n\n"

        async def generate():
            if avviso_contesto and not context:
                yield f'data: {json.dumps({"token": avviso_contesto})}\n\n'
                yield f'data: [DONE]\n\n'
                return

            payload = {
                "model": "qwen3.5:9b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Sei il pianificatore ufficiale di itinerari di TPL FVG. "
                            "Il tuo compito è costruire un itinerario personalizzato basandoti "
                            "ESCLUSIVAMENTE sulle tabelle orarie fornite nel contesto.\n\n"
                            "Regole rigide:\n"
                            "1. Identifica la fermata di partenza e di destinazione chieste dall'utente.\n"
                            "2. Cerca tra le linee fornite nel contesto quella (o quelle) che collegano i due punti.\n"
                            "3. Se trovi una soluzione, rispondi strutturando l'itinerario in modo chiaro.\n"
                            "4. Se le località o le combinazioni NON sono presenti nel contesto, NON inventare "
                            "itinerari stradali, treni o bus fittizi."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Ecco i dati orari:\n\n{context}\n\nRichiesta Utente: {message}"
                    }
                ],
                "stream": True,
                "options": {"temperature": 0.1, "num_ctx": 4096}
            }
            try:
                resp = _requests.post("http://localhost:11434/api/chat", json=payload, stream=True, timeout=120)
                for line in resp.iter_lines():
                    if line:
                        chunk_data = json.loads(line)
                        token = chunk_data.get("message", {}).get("content", "")
                        if token:
                            yield f'data: {json.dumps({"token": token})}\n\n'
                        if chunk_data.get("done"):
                            break
                yield f'data: [DONE]\n\n'
            except Exception as e:
                yield f'data: {json.dumps({"error": str(e)})}\n\n'

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


def step4_avvia_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    _titolo("STEP 4 — FastAPI + Uvicorn")
    db_ok = _stato_pipeline.get("vectorstore") is not None
    print(f"  DB vettoriale : {'✓ pronto' if db_ok else '✗ NON disponibile'}")
    print(f"  Embedding     : nomic-embed-text:latest")
    print(f"  LLM           : qwen3.5:9b")
    print(f"\n  🌐 INTERFACCIA INTERATTIVA WEB: http://{host}:{port}/")
    app = _crea_app()
    uvicorn.run(app, host=host, port=port)


def _titolo(testo: str): print(f"\n{'='*60}\n  {testo}\n{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",  default="trasporti_generato.json",
                        help="Percorso del file JSON orari da generare/caricare")
    parser.add_argument("--model", default="qwen3.5:9b",
                        help="Modello LLM da usare su Ollama per la Chat")
    args = parser.parse_args()

    _stato_pipeline["llm_model"] = args.model
    _stato_pipeline["json_path"] = args.path

    if not step1_test_ollama(args.model):
        sys.exit(1)

    # Richiama il nuovo step di generazione del file JSON
    step0_genera_json_strutturato(args.path)

    chunks = step2_ingestione(args.path)
    _stato_pipeline["chunks"] = chunks
    step3_vettoriale(chunks)

    step4_avvia_server()


if __name__ == "__main__":
    main()