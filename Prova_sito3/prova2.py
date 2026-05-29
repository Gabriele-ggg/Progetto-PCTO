"""
RAG ENGINE — Sistema RAG Locale da PDF con Ollama

Implementazione completa in un unico script Python senza utilizzare classi.

Approccio procedurale:
- funzioni modulari
- variabili globali per vector store
- pipeline lineare
- compatibilità massima con Python CLI

Architettura aggiornata:
- Embedding Model: Ollama mxbai-embed-large:latest
- LLM QA Model: Ollama llama3:8b
- Vector Store: NumPy in-memory
- Retrieval: Cosine Similarity
- Parsing PDF: pypdf
- Tutto completamente locale

Requisiti implementati dal PDF:

FR-01 → Scansione directory PDF
FR-02 → Chunking con overlap 10%
FR-03 → Embedding semantici locali
FR-04 → Database vettoriale NumPy
FR-05 → Query terminale con risposta AI
NFR-01 → Nessun cloud
NFR-02 → Similarità cosinica veloce
NFR-03 → Gestione PDF corrotti

Funzionalità implementate:
- FR-01: Ingestion e Parsing PDF
- FR-02: Text Chunking con overlap
- FR-03: Generazione Embedding Semantici
- FR-04: Database Vettoriale In-Memory
- FR-05: Query & QA da terminale
- NFR-01: Tutto locale
- NFR-02: Ricerca veloce con NumPy
- NFR-03: Gestione errori PDF

Dipendenze:
    pip install pypdf numpy requests

Struttura progetto:
rag_terminal_app/
├── data/
│   └── documenti.pdf
└── rag_engine.py
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
from pypdf import PdfReader
import requests


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATA_DIR = Path("data")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"
OLLAMA_LLM_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# PDF INGESTION
# ============================================================


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Estrae il testo da tutte le pagine di un PDF.
    """

    extracted_text = []

    try:
        print("[INFO] Inizializzazione RAG Engine...")
        print(f"[EMBEDDING MODEL] {OLLAMA_EMBEDDING_MODEL}")
        print(f"[LLM MODEL] {OLLAMA_LLM_MODEL}
")

        print("
[INFO] Costruzione indice vettoriale...")

        build_index()

        print("\n[SUCCESS] Sistema pronto.")
        print("Digita una domanda oppure 'exit' per uscire.\n")

        while True:
            question = input("Domanda > ").strip()

            if question.lower() in ["exit", "quit"]:
                print("\n[INFO] Chiusura applicazione.")
                break

            if not question:
                continue

            print("\n[INFO] Ricerca semantica in corso...")

            result = answer_question(question)

            print("\n" + "-" * 70)
            print("RISPOSTA")
            print("-" * 70)
            print(result["answer"])

            print("
Risposta generata da Llama3 tramite Ollama")

            print("\nChunk utilizzati:")

            for idx, source in enumerate(result["sources"], start=1):
                chunk = source["chunk"]

                print(f"\n[{idx}] Documento: {chunk['document']}")
                print(f"Similarity Score: {source['score']:.4f}")
                print(f"Chunk ID: {chunk['chunk_id']}")
                print("Anteprima:")
                print(chunk["text"][:250].replace("\n", " "))

            print("\n")

    except Exception as error:
        print(f"\n[FATAL ERROR] {error}")


if __name__ == "__main__":
    main()
