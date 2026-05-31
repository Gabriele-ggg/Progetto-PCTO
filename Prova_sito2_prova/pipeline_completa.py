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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

# ============================================================
# PDF EXTRACTION & PARSING UTILITIES - OTTIMIZZATE
# ============================================================

_PDF_CACHE = {}  # Cache per PDF già processati
_CHUNK_CACHE = set()  # Cache di hash chunk per deduplicazione

def _hash_testo(testo: str) -> str:
    """Genera hash MD5 di un testo per deduplicazione."""
    return hashlib.md5(testo.encode()).hexdigest()

@lru_cache(maxsize=128)
def _estrai_testo_da_pdf(percorso_pdf: str) -> str:
    """Estrae il testo da un file PDF usando pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(percorso_pdf)
        testo = ""
        for pagina in reader.pages:
            testo += pagina.extract_text() or ""
        return testo
    except ImportError:
        print(f"  [ERRORE] pypdf non installato. Installa con: pip install pypdf")
        return ""
    except Exception as e:
        print(f"  [ERRORE] Lettura PDF {percorso_pdf}: {e}")
        return ""


def _estrai_linee_da_testo(testo: str, tipo: str = "urbani") -> dict:
    """
    Estrae linee, percorsi, fermate e orari dal testo PDF.
    Ritorna un dizionario con i dati strutturati.
    """
    linee_dict = {}
    
    # Regex per identificare numeri di linea (es. "Linea 1", "Line 1", "1", "81")
    pattern_linea = r'(?:Linea|Line|N\.?)\s*([0-9A-Z]+)(?:\s|:|$|\.)'
    
    # Regex per orari (es. "06:10", "6:10", "06.10")
    pattern_orario = r'([0-2]?[0-9][:\.][0-5][0-9])'
    
    # Suddividi il testo in sezioni per linea
    sezioni = re.split(r'(?:Linea|Line|N\.?)\s+', testo, flags=re.IGNORECASE)
    
    for sezione in sezioni[1:]:  # Salta la prima sezione vuota
        linee = re.findall(r'^([0-9A-Z]+)', sezione)
        if not linee:
            continue
        
        nome_linea = linee[0].strip()
        
        # Estrai il contenuto della linea (escluendo il numero)
        contenuto = re.sub(r'^[0-9A-Z]+\s*', '', sezione, count=1)
        
        # Dividi in andata e ritorno
        linee_sezioni = re.split(r'(?:Ritorno|Return|Andata|Outbound|Inverso)', contenuto, flags=re.IGNORECASE)
        
        # Estrai orari
        orari_trovati = re.findall(pattern_orario, contenuto)
        orari_unici = sorted(list(set([o.replace('.', ':') for o in orari_trovati])))
        
        # Estrai fermate (linee che iniziano con lettere maiuscole o contengono "via", "piazza", "viale")
        fermate = []
        for linea_text in contenuto.split('\n'):
            linea_text = linea_text.strip()
            # Cerca pattern di fermate
            if any(keyword in linea_text.lower() for keyword in ['via ', 'piazza', 'viale', 'corso', 'ponte', 'stazione', 'fs', 'centrale']):
                # Rimuovi gli orari dalla linea di fermata
                fermata_name = re.sub(pattern_orario, '', linea_text).strip()
                if fermata_name and len(fermata_name) > 2:
                    fermate.append(fermata_name[:80])  # Limita lunghezza
        
        fermate_unique = []
        for f in fermate:
            if f not in fermate_unique:
                fermate_unique.append(f)
        
        # Crea struttura linea
        if nome_linea not in linee_dict:
            linee_dict[nome_linea] = {
                "tipo": tipo,
                "andata": {
                    "percorso": " - ".join(fermate_unique[:5]) if fermate_unique else "N/D",
                    "fermate": fermate_unique[:10] if fermate_unique else [],
                    "orari": orari_unici[:20] if orari_unici else []
                },
                "ritorno": {
                    "percorso": " - ".join(reversed(fermate_unique[:5])) if fermate_unique else "N/D",
                    "fermate": list(reversed(fermate_unique[:10])) if fermate_unique else [],
                    "orari": orari_unici[:20] if orari_unici else []
                }
            }
    
    return linee_dict


def _estrai_dati_da_cartella_pdf(cartella_base: str = "./pdf", max_workers: int = 4) -> dict:
    """
    Estrae dati da tutti i PDF nelle cartelle urbani/ e extraurbani/ con parallelizzazione.
    Ritorna un dizionario con tutte le linee estratte.
    """
    print("  Ricerca PDF nelle cartelle...")
    
    tutte_le_linee = {}
    
    # Cartelle supportate
    cartelle = {
        "urbani": "urbano",
        "extraurbani": "extraurbano"
    }
    
    base_path = Path(cartella_base)
    
    # Raccogli tutti i PDF da processare
    pdf_tasks = []
    for cartella_nome, tipo_linea in cartelle.items():
        cartella_path = base_path / cartella_nome
        
        if not cartella_path.exists():
            print(f"    [AVVISO] Cartella non trovata: {cartella_path}")
            continue
        
        pdf_files = list(cartella_path.glob("*.pdf"))
        print(f"    [{len(pdf_files)} PDF] trovati in {cartella_nome}/")
        
        for pdf_file in pdf_files:
            pdf_tasks.append((str(pdf_file), pdf_file.name, tipo_linea))
    
    # Processa PDF in parallelo con ThreadPoolExecutor
    if pdf_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(_estrai_testo_da_pdf, path): (name, tipo) 
                for path, name, tipo in pdf_tasks
            }
            
            for future in as_completed(future_to_pdf):
                pdf_name, tipo_linea = future_to_pdf[future]
                try:
                    testo = future.result()
                    if testo:
                        linee = _estrai_linee_da_testo(testo, tipo_linea)
                        tutte_le_linee.update(linee)
                        print(f"        [✓] Estratte {len(linee)} linee da {pdf_name}")
                    else:
                        print(f"        [⚠] Nessun testo estratto da {pdf_name}")
                except Exception as e:
                    print(f"        [✗] Errore processing {pdf_name}: {e}")
    
    return tutte_le_linee


def step0_genera_json_strutturato(path_output: str = "trasporti_generato.json", pdf_folder: str = "./pdf", max_workers: int = 4):
    """
    Genera un file JSON con la stessa esatta struttura del file 
    'trasporti_non_copiare_prendere_spunto.json', leggendo i dati dai PDF
    nelle cartelle urbani/ e extraurbani/.
    """
    _titolo("STEP 0 — Generazione JSON Strutturato da PDF")
    
    # Estrai dati dai PDF (supporta parallelizzazione via max_workers)
    linee_estratte = _estrai_dati_da_cartella_pdf(pdf_folder, max_workers=max_workers)
    
    if not linee_estratte:
        print("  [AVVISO] Nessuna linea estratta dai PDF. Creazione file JSON con struttura di default...")
        linee_estratte = {
            "1": {
                "tipo": "urbano",
                "andata": {
                    "percorso": "Via Chiusaforte-Ospedale-Chiavris-1°Maggio-FS-Gervasutta",
                    "fermate": ["Via Chiusaforte", "Ospedale", "Chiavris", "1°Maggio", "FS", "Gervasutta"],
                    "orari": ["06:10", "06:25", "06:36", "06:54", "07:05", "07:20", "07:35", "07:50"]
                },
                "ritorno": {
                    "percorso": "Gervasutta-FS-1°Maggio-Chiavris-Ospedale-via Chiusaforte",
                    "fermate": ["Gervasutta", "FS", "1°Maggio", "Chiavris", "Ospedale", "Via Chiusaforte"],
                    "orari": ["06:15", "06:30", "06:45", "07:00", "07:15", "07:30", "07:45", "08:00"]
                }
            }
        }
    
    # Costruisci la struttura JSON finale
    linee_array = []
    
    for nome_linea in sorted(linee_estratte.keys(), key=lambda x: (x.isalpha(), x)):
        dati_linea = linee_estratte[nome_linea]
        
        linea_obj = {
            "n": nome_linea,
            "tipo": dati_linea.get("tipo", "urbano"),
            "a": {
                "p": dati_linea.get("andata", {}).get("percorso", "N/D"),
                "f": _crea_fermate(dati_linea.get("andata", {}).get("fermate", []), 
                                   dati_linea.get("andata", {}).get("orari", []))
            }
        }
        
        # Aggiungi ritorno se disponibile
        if dati_linea.get("ritorno", {}).get("fermate"):
            linea_obj["r"] = {
                "p": dati_linea.get("ritorno", {}).get("percorso", "N/D"),
                "f": _crea_fermate(dati_linea.get("ritorno", {}).get("fermate", []), 
                                   dati_linea.get("ritorno", {}).get("orari", []))
            }
        
        linee_array.append(linea_obj)
    
    # Data odierna
    data_oggi = datetime.now().strftime("%Y-%m-%d")
    
    struttura_json = {
        "servizio": "Servizio Urbano e Extraurbano",
        "orario_dal": data_oggi,
        "linee": linee_array
    }
    
    try:
        with open(path_output, "w", encoding="utf-8") as f:
            json.dump(struttura_json, f, ensure_ascii=False, indent=4)
        print(f"  [✓] File JSON generato con successo e salvato in: {path_output}")
        print(f"  [✓] Linee estratte: {len(linee_array)}")
        return path_output
    except Exception as e:
        print(f"  [ERRORE] Impossibile generare il file JSON: {e}")
        return None


def _crea_fermate(nomi_fermate: list, orari: list) -> list:
    """
    Crea la struttura delle fermate con orari per il JSON.
    Ogni fermata avrà codice, nome, città e orari.
    """
    fermate_strutturate = []
    
    for idx, nome_fermata in enumerate(nomi_fermate):
        fermata_obj = {
            "c": f"UD{100 + idx:03d}",  # Codice fermata
            "n": nome_fermata,  # Nome fermata
            "ci": "UDINE",  # Città
            "v": nome_fermata,  # Via
            "o": orari if orari else []  # Orari
        }
        fermate_strutturate.append(fermata_obj)
    
    return fermate_strutturate


# ============================================================
# STEP 1 — Test connessione Ollama
# ============================================================

def step1_test_ollama(model: str = "llama3:8b") -> bool:
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
    _titolo("STEP 2 — Ingestione + Chunking (OTTIMIZZATO)")
    t_start = time.time()

    print(f"  Caricamento file JSON: {path}")
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
    
    # Deduplicazione chunk
    print(f"  Deduplicazione chunk...")
    unique_chunks = []
    seen_hashes = set()
    
    for chunk in chunks:
        chunk_hash = _hash_testo(chunk.page_content)
        if chunk_hash not in seen_hashes:
            unique_chunks.append(chunk)
            seen_hashes.add(chunk_hash)
    
    print(f"  [✓] {len(chunks)} chunk → {len(unique_chunks)} unici in {time.time() - t_start:.2f}s")
    print(f"  [ℹ] Riduzione: {len(chunks) - len(unique_chunks)} duplicati rimossi")
    return unique_chunks


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
# STEP 3 — Vector DB (ChromaDB Embeddings) - OTTIMIZZATO
# ============================================================

CHROMA_BASE = "./chroma_db"

def step3_vettoriale(chunks: list, model: str = "nomic-embed-text:latest", batch_size: int = 100):
    """Carica i chunk nel database vettoriale con ottimizzazioni massime.
    
    Ottimizzazioni:
    - Incremental indexing (aggiorna DB esistente)
    - Deduplica automaticamente nei chunk
    - Batch processing efficiente
    - Caching persistente
    """
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_chroma import Chroma
    import os

    _titolo("STEP 3 — Vector DB (ChromaDB) - SUPER OTTIMIZZATO")

    if not chunks:
        print("  [ATTENZIONE] Nessun chunk da indicizzare. Step 3 saltato.")
        return

    model_short = "nomic" if "nomic" in model else "mxbai"
    persist_dir = f"{CHROMA_BASE}_{model_short}"
    
    # Verifica se il database esiste già
    db_exists = os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    
    try:
        print(f"  Configurazione embedding: {model}")
        print(f"  Batch size: {batch_size}")
        print(f"  Totale chunk: {len(chunks)}")
        
        # Inizializza embeddings
        embeddings = OllamaEmbeddings(
            model=model,
            base_url="http://localhost:11434",
            show_progress=True
        )
        
        # Se il DB esiste, usa il metodo incremental
        if db_exists and len(chunks) > 0:
            print(f"  [ℹ] Database esistente trovato. Update incrementale...")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
                collection_name=f"docs_{model_short}",
            )
            
            # Filtra chunk già presenti usando hash
            existing_hashes = set()
            if hasattr(vectorstore, '_collection'):
                for item in vectorstore._collection.get():
                    if item and 'documents' in item:
                        for doc in item['documents']:
                            existing_hashes.add(_hash_testo(doc))
            
            new_chunks = [c for c in chunks if _hash_testo(c.page_content) not in existing_hashes]
            
            if new_chunks:
                print(f"  [ℹ] Aggiunta {len(new_chunks)} nuovi chunk...")
                # Processa in batch
                for i in range(0, len(new_chunks), batch_size):
                    batch = new_chunks[i:i+batch_size]
                    vectorstore.add_documents(batch)
                    print(f"      Batch {i//batch_size + 1}/{(len(new_chunks)-1)//batch_size + 1} completato")
                print(f"  [✓] {len(new_chunks)} chunk aggiunti (skipped {len(chunks)-len(new_chunks)} duplicati)")
            else:
                print(f"  [ℹ] Nessun nuovo chunk da aggiungere (tutti già presenti)")
        else:
            print(f"  [ℹ] Creazione nuovo database vettoriale...")
            # Processa in batch anche per la creazione
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                if i == 0:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                        collection_name=f"docs_{model_short}",
                    )
                else:
                    vectorstore.add_documents(batch)
                print(f"  Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} creato")
        
        # Persisti i cambiamenti
        vectorstore.persist()
        _stato_pipeline["vectorstore"] = vectorstore
        _stato_pipeline["embedding_model"] = model
        
        # Stampa statistiche
        print(f"  [✓] Database vettoriale pronto con {model}")
        print(f"  [✓] Percorso: {persist_dir}")
        print(f"  [✓] Memoria persistente salvata")
        
    except Exception as e:
        print(f"  [⚠ FALLBACK] Errore con {model}: {e}")
        if "mxbai" in model:
            print(f"  Tentativo con fallback: nomic-embed-text:latest")
            step3_vettoriale(chunks, model="nomic-embed-text:latest", batch_size=batch_size)
        else:
            print(f"  [ERRORE CRITICO] Verifica che Ollama sia running: ollama serve")

# ============================================================
# STEP 4 — Server FastAPI & Endpoints
# ============================================================

_stato_pipeline: dict = {
    "chunks": [],
    "vectorstore": None,
    "llm_model": "qwen3.5:9b",
    "json_path": "trasporti_generato.json",
    "embedding_model": "nomic-embed-text:latest",
    "batch_size": 100,
    "query_cache": {},  # Cache query → risultati
}

def _crea_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    import requests as _requests

    app = FastAPI()
    static_dir = Path(__file__).resolve().parent
    index_file = static_dir / "index.html"
    
    # Funzione helper per query ottimizzate
    def _search_vectorstore_cached(message: str, k: int = 6, use_cache: bool = True):
        """Ricerca nel vectorstore con caching opzionale."""
        cache_key = f"{message}:{k}"
        query_cache = _stato_pipeline.get("query_cache", {})
        
        # Controlla cache
        if use_cache and cache_key in query_cache:
            return query_cache[cache_key]
        
        # Esegui ricerca
        vs = _stato_pipeline.get("vectorstore")
        if not vs:
            return None
        
        results = vs.similarity_search_with_score(message, k=k)
        
        # Salva in cache (max 1000 query)
        if len(query_cache) < 1000:
            query_cache[cache_key] = results
        
        return results

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
            "embedding": _stato_pipeline.get("embedding_model"),
            "batch_size": _stato_pipeline.get("batch_size"),
            "llm": _stato_pipeline.get("llm_model"),
            "cache_queries": len(_stato_pipeline.get("query_cache", {})),
        }

    @app.post("/api/reload")
    def frontend_reload():
        try:
            path_json = _stato_pipeline.get("json_path", "trasporti_generato.json")
            pdf_folder = _stato_pipeline.get("pdf_folder", "./pdf")
            embedding_model = _stato_pipeline.get("embedding_model", "nomic-embed-text:latest")
            batch_size = _stato_pipeline.get("batch_size", 100)
            
            # Pulisci cache
            _stato_pipeline["query_cache"] = {}
            
            # Prima rigenereiamo il JSON strutturato dai PDF
            step0_genera_json_strutturato(path_output=path_json, pdf_folder=pdf_folder)
            # Poi eseguiamo l'ingestione
            chunks = step2_ingestione(path=path_json)
            _stato_pipeline["chunks"] = chunks
            # Infine vettorizziamo con il modello configurato
            step3_vettoriale(chunks, model=embedding_model, batch_size=batch_size)
            db_ok = _stato_pipeline.get("vectorstore") is not None
            return {
                "status": "success" if db_ok else "warning",
                "chunks": len(chunks),
                "db_pronto": db_ok,
                "embedding_model": embedding_model,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat")
    async def frontend_chat(data: dict):
        message = data.get("message", "").strip()

        if not message:
            return {"error": "Messaggio vuoto"}

        context = ""
        avviso_contesto = ""
        
        try:
            # Usa ricerca ottimizzata con cache
            res = _search_vectorstore_cached(message, k=5, use_cache=True)
            
            if res:
                # Seleziona i risultati migliori
                for doc, score in res:
                    if score < 1.0:  # Filtra score basso
                        context += f"{doc.page_content}\n---\n"
            else:
                avviso_contesto = "[Nessun dato rilevante trovato nel database per questa richiesta.]\n\n"
                
        except Exception as e:
            avviso_contesto = f"[Errore nel recupero del contesto: {e}]\n\n"

        async def generate():
            if avviso_contesto and not context:
                yield f'data: {json.dumps({"token": avviso_contesto})}\n\n'
                yield f'data: [DONE]\n\n'
                return

            payload = {
                "model": _stato_pipeline.get("llm_model", "qwen3.5:9b"),
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
                            "4. Se le località o le combinazioni NON sono presenti nel contesto, NON inventare itinerari."
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
    emb_model = _stato_pipeline.get("embedding_model", "nomic-embed-text:latest")
    print(f"  DB vettoriale : {'✓ pronto' if db_ok else '✗ NON disponibile'}")
    print(f"  Embedding     : {emb_model}")
    print(f"  Batch size    : {_stato_pipeline.get('batch_size', 100)}")
    print(f"  LLM           : {_stato_pipeline.get('llm_model', 'qwen3.5:9b')}")
    print(f"  Cache query   : {len(_stato_pipeline.get('query_cache', {}))} entries")
    print(f"\n  🌐 INTERFACCIA INTERATTIVA WEB: http://{host}:{port}/")
    app = _crea_app()
    uvicorn.run(app, host=host, port=port)


def _titolo(testo: str): print(f"\n{'='*60}\n  {testo}\n{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",      default="trasporti_generato.json",
                        help="Percorso del file JSON orari da generare/caricare")
    parser.add_argument("--pdf",       default="./pdf",
                        help="Cartella base contenente le sottocartelle urbani/ e extraurbani/ con i PDF")
    parser.add_argument("--model",     default="qwen3.5:9b",
                        help="Modello LLM da usare su Ollama per la Chat")
    parser.add_argument("--embedding", default="nomic-embed-text:latest",
                        choices=["nomic-embed-text:latest", "mxbai-embed-large:latest"],
                        help="Modello embedding: nomic (veloce) o mxbai (preciso)")
    parser.add_argument("--batch",     type=int, default=100,
                        help="Dimensione batch per embedding processing")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Numero di worker per parallelizzazione PDF (default: 4)")
    parser.add_argument("--port",      type=int, default=8000,
                        help="Porta per FastAPI (default: 8000)")
    args = parser.parse_args()

    # Stampa config iniziale
    print("\n" + "="*60)
    print(" PIPELINE TPL-FVG — SISTEMA COMPLETO OTTIMIZZATO")
    print("="*60)
    print(f"  📁 PDF Folder: {args.pdf}")
    print(f"  📄 JSON Path: {args.path}")
    print(f"  🧠 Embedding Model: {args.embedding}")
    print(f"  📦 Batch Size: {args.batch}")
    print(f"  ⚙️  PDF Workers: {args.workers}")
    print(f"  🌐 API Port: {args.port}")
    print("="*60 + "\n")

    _stato_pipeline["llm_model"] = args.model
    _stato_pipeline["json_path"] = args.path
    _stato_pipeline["pdf_folder"] = args.pdf
    _stato_pipeline["embedding_model"] = args.embedding
    _stato_pipeline["batch_size"] = args.batch
    _stato_pipeline["workers"] = args.workers

    if not step1_test_ollama(args.model):
        sys.exit(1)

    # Esecuzione pipeline completa con parallelizzazione
    step0_genera_json_strutturato(path_output=args.path, pdf_folder=args.pdf, max_workers=args.workers)
    chunks = step2_ingestione(args.path)
    _stato_pipeline["chunks"] = chunks
    step3_vettoriale(chunks, model=args.embedding, batch_size=args.batch)

    # Avvia server
    print(f"\n🚀 Avviamento server FastAPI...")
    print(f"🌐 http://localhost:{args.port}/")
    print(f"📡 WebSocket: ws://localhost:{args.port}/ws\n")
    step4_avvia_server(port=args.port)


if __name__ == "__main__":
    main()