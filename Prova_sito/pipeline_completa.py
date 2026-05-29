"""
pipeline_completa.py — Edizione Itinerari Personalizzati
- Risolto NameError (documentos -> documenti)
- Implementato l'endpoint /api/reload e completato /api/status per index.html
- Sistema di Routing RAG potenziato (k=6) con System Prompt per calcolo itinerari
"""

import argparse
import json
import time
import sys
import os
import re
from pathlib import Path

# ============================================================
# PARSING PDF (Euristico)
# ============================================================

import json
import re
import requests
from pathlib import Path

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
        "}"
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

def _estrai_corse_da_pdf(pdf_file, tipologia):
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


def _ollama_in_esecuzione() -> bool:
    import requests
    try:
        requests.get("http://localhost:11434", timeout=1)
        return True
    except requests.RequestException:
        return False


def _avvia_ollama_automaticamente() -> bool:
    import subprocess

    if _ollama_in_esecuzione():
        print("  [OK] Ollama già in esecuzione")
        return True

    print("  Avvio automatico di Ollama con 'ollama serve'...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    except FileNotFoundError:
        print("  [ERRORE] comando 'ollama' non trovato nel PATH")
        return False
    except OSError as e:
        print(f"  [ERRORE] impossibile avviare Ollama: {e}")
        return False

    for _ in range(30):
        time.sleep(1)
        if _ollama_in_esecuzione():
            print("  [OK] Ollama avviato automaticamente")
            return True

    print("  [ERRORE] timeout nell'avvio automatico di Ollama")
    return False


# ============================================================
# STEP 1 — Test connessione Ollama
# ============================================================

def step1_test_ollama(model: str = "qwen2.5:1.5b") -> bool:
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
# STEP 2 — Ingestione e chunking
# ============================================================

def step2_ingestione(source: str = "json", path: str = "trasporti.json") -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _titolo("STEP 2 — Ingestione + Chunking")
    t_start = time.time()

    if source == "pdf":
        print("  Conversione PDF in JSON strutturato...")
        base_path = Path(__file__).resolve().parent / "pdf"
        
        pdf_tasks = []
        for tipologia in ["urbani", "extraurbani"]:
            cartella = base_path / tipologia
            if cartella.is_dir():
                pdf_files = sorted(cartella.glob("*.pdf"))
                print(f"  Trovati {len(pdf_files)} file in '{tipologia}'")
                pdf_tasks.extend([(f, tipologia) for f in pdf_files])
        
        full_json = {"linee": []}
        for idx, (pdf_file, tipologia) in enumerate(pdf_tasks):
            linea_data = _estrai_corse_da_pdf(pdf_file, tipologia)
            if linea_data and linea_data["corse"]:
                full_json["linee"].append(linea_data)
                
        json_path = base_path.parent / "trasporti_generato.json"
        with open(json_path, "w", encoding="utf-8") as out_f:
            json.dump(full_json, out_f, indent=2, ensure_ascii=False)
            
        documenti = _carica_json(str(json_path))
    else:
        print(f"  Caricamento JSON: {path}")
        documenti = _carica_json(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documenti)
    print(f"  [✓] {len(chunks)} chunk generati in {time.time() - t_start:.2f}s")
    return chunks

def _carica_json(path: str):
    from langchain_core.documents import Document
    try:
        with open(path, encoding="utf-8") as f:
            dati = json.load(f)
    except Exception as e:
        print(f"    [ERRORE] Lettura JSON in {path}: {e}")
        return []
    
    documenti = []
    for linea in dati.get("linee", []):
        nome_linea = str(linea.get("nome", "N/D"))
        direzione = str(linea.get("direzione", "N/D"))
        
        for corsa in linea.get("corse", []):
            fermate_testo = "\n".join(f"{f['orario']} — {f['nome']}" for f in corsa.get("fermate", []))
            testo = (
                f"Linea: {nome_linea} ({linea.get('tipo', 'N/D')})\n"
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
                    "tipologia": linea.get("tipo", "N/D")
                }
            ))
    return documenti  # FISSO: Corretto il NameError da 'documentos' a 'documenti'

# ============================================================
# STEP 3 — Vector DB (ChromaDB)
# ============================================================

MODELLI_EMBEDDING = {"nomic": "nomic-embed-text", "mxbai": "mxbai-embed-large"}
CHROMA_BASE = "./chroma_db"

def step3_vettoriale(chunks: list):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_chroma import Chroma
    
    for nome, model_id in MODELLI_EMBEDDING.items():
        persist_dir = f"{CHROMA_BASE}_{nome}"
        try:
            embeddings = OllamaEmbeddings(model=model_id)
            if os.path.exists(persist_dir):
                # Pulizia della vecchia istanza se necessario ricaricare
                vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name=f"docs_{nome}")
            else:
                vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir, collection_name=f"docs_{nome}")
            _stato_pipeline[f"vectorstore_{nome}"] = vectorstore
            print(f"  [✓] Database vettoriale {nome} pronto.")
        except Exception as e:
            print(f"  [ERRORE] {nome}: {e}")

# ============================================================
# STEP 4 — Server FastAPI & Endpoints
# ============================================================

_stato_pipeline: dict = {
    "chunks": [],
    "vectorstore_nomic": None,
    "vectorstore_mxbai": None,
    "llm_model": "qwen2.5:1.5b"
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
            linea = meta.get("linea", "")
            tipologia = meta.get("tipologia", "")
            corsa_id = meta.get("corsa", "")
            
            if linea:
                linee_set[linea] = tipologia
            if tipologia:
                tipi_count[tipologia] = tipi_count.get(tipologia, 0) + 1
            if corsa_id:
                corse_set.add(corsa_id)
        
        linee_list = [{"linea": l, "tipo": t} for l, t in linee_set.items()]
        linee_list.sort(key=lambda x: x["linea"])
        
        pdf_dir = static_dir / "pdf"
        pdf_analizzati = [f.name for f in pdf_dir.glob("**/*.pdf")] if pdf_dir.exists() else []

        return {
            "totale_corse": len(corse_set),
            "totale_linee": len(linee_set),
            "pdf_analizzati": pdf_analizzati,
            "generato_il": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tipi": tipi_count,
            "linee": linee_list
        }

    @app.post("/api/reload")
    def frontend_reload():
        try:
            pdf_dir = static_dir / "pdf"
            source = "pdf" if (pdf_dir.exists() and any(pdf_dir.glob("**/*.pdf"))) else "json"
            chunks = step2_ingestione(source=source, path="trasporti.json")
            _stato_pipeline["chunks"] = chunks
            step3_vettoriale(chunks)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat")
    async def frontend_chat(data: dict):
        message = data.get("message", "")
        context = ""
        try:
            # Usiamo nomic o mxbai a seconda di quale sia attivo nello stato
            vs = _stato_pipeline.get("vectorstore_nomic") or _stato_pipeline.get("vectorstore_mxbai")
            if vs:
                # k=6 aumenta la probabilità di catturare partenza, destinazione e coincidenze insieme
                res = vs.similarity_search_with_score(message, k=6)
                if res:
                    for doc, _ in res:
                        context += f"{doc.page_content}\n---\n"
        except Exception as e:
            print(f"Errore RAG: {e}")
        
        async def generate():
            payload = {
                "model": _stato_pipeline.get("llm_model", "qwen2.5:1.5b"),
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
                            "3. Se trovi una soluzione, rispondi strutturando l'itinerario in modo chiaro:\n"
                            "   - Linea utilizzata e Direzione\n"
                            "   - Orario esatto di partenza dalla fermata iniziale\n"
                            "   - Eventuali fermate intermedie rilevanti e l'orario di arrivo finale.\n"
                            "4. Se l'utente chiede un orario specifico, trova la corsa che si avvicina meglio.\n"
                            "5. Se le località o le combinazioni NON sono presenti nel contesto, NON inventare "
                            "itinerari stradali, treni o bus fittizi. Di' chiaramente che non ci sono corse ufficiali "
                            "caricate nel sistema che collegano le due posizioni."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Ecco i dati orari estratti dal database:\n\n{context}\n\nRichiesta Utente: {message}"
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
                        if chunk_data.get("done"): break
                yield f'data: [DONE]\n\n'
            except Exception as e:
                yield f'data: {json.dumps({"error": str(e)})}\n\n'
        
        return StreamingResponse(generate(), media_type="text/event-stream")

    return app

def step4_avvia_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    _titolo("STEP 4 — FastAPI + Uvicorn")
    print(f"\n  🌐 INTERFACCIA INTERATTIVA WEB: http://{host}:{port}/")
    app = _crea_app()
    uvicorn.run(app, host=host, port=port)

def _titolo(testo: str): print(f"\n{'='*60}\n  {testo}\n{'='*60}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["pdf", "json"], default="json")
    parser.add_argument("--path", default="trasporti.json")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args()

    _stato_pipeline["llm_model"] = args.model
    if not _avvia_ollama_automaticamente():
        sys.exit(1)
    if not step1_test_ollama(args.model):
        sys.exit(1)
        
    chunks = step2_ingestione(args.source, args.path)
    _stato_pipeline["chunks"] = chunks
    step3_vettoriale(chunks)
    step4_avvia_server()

if __name__ == "__main__":
    main()