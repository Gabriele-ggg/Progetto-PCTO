```python
import requests
import numpy as np

from pathlib import Path
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATA_DIR = Path("data")

EMBEDDING_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3:8b"

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"

TOP_K = 3


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL,
)


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


# ============================================================
# PDF PARSER
# ============================================================

def estrai_testo_pdf(pdf_path: Path) -> str:

    testo_completo = []

    try:
        reader = PdfReader(str(pdf_path))

        for numero_pagina, pagina in enumerate(reader.pages, start=1):

            try:
                testo = pagina.extract_text()

                if testo:
                    testo_completo.append(testo)

                else:
                    print(
                        f"[WARNING] Nessun testo nella pagina "
                        f"{numero_pagina} di '{pdf_path.name}'"
                    )

            except Exception as errore_pagina:

                print(
                    f"[ERROR] Errore pagina {numero_pagina}: "
                    f"{errore_pagina}"
                )

    except Exception as errore_pdf:

        print(
            f"[ERROR] Impossibile leggere "
            f"'{pdf_path.name}': {errore_pdf}"
        )

        return ""

    return "\n".join(testo_completo)


# ============================================================
# SCANSIONE DIRECTORY PDF
# ============================================================

def carica_documenti():

    documenti = []

    if not DATA_DIR.exists():

        print(f"[ERROR] Directory '{DATA_DIR}' non trovata.")
        return documenti

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:

        print("[INFO] Nessun PDF trovato.")
        return documenti

    for pdf_file in pdf_files:

        print(f"[INFO] Lettura PDF: {pdf_file.name}")

        testo = estrai_testo_pdf(pdf_file)

        if testo.strip():

            documenti.append(
                {
                    "nome_file": pdf_file.name,
                    "testo": testo,
                }
            )

            print(
                f"[SUCCESS] Estratti "
                f"{len(testo)} caratteri"
            )

    return documenti


# ============================================================
# CHUNKING
# ============================================================

def crea_chunks(documenti):

    chunks_finali = []

    for documento in documenti:

        chunks = text_splitter.split_text(
            documento["testo"]
        )

        for indice, chunk in enumerate(chunks):

            chunks_finali.append(
                {
                    "documento": documento["nome_file"],
                    "chunk_id": indice,
                    "testo": chunk,
                }
            )

    return chunks_finali


# ============================================================
# EMBEDDING OLLAMA
# ============================================================

def genera_embedding(testo):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": testo,
        },
        timeout=120,
    )

    response.raise_for_status()

    embedding = response.json()["embedding"]

    vettore = np.array(
        embedding,
        dtype=np.float32,
    )

    # normalizzazione
    vettore = vettore / np.linalg.norm(vettore)

    return vettore


# ============================================================
# CREAZIONE VECTOR STORE
# ============================================================

def indicizza_chunks(chunks):

    embeddings = []

    for indice, chunk in enumerate(chunks, start=1):

        print(
            f"[EMBEDDING] "
            f"{indice}/{len(chunks)}"
        )

        embedding = genera_embedding(
            chunk["testo"]
        )

        embeddings.append(embedding)

    matrice_embeddings = np.vstack(embeddings)

    return matrice_embeddings


# ============================================================
# RICERCA SEMANTICA
# ============================================================

def ricerca_semantica(
    domanda,
    chunks,
    matrice_embeddings,
    top_k=TOP_K,
):

    embedding_domanda = genera_embedding(domanda)

    similarita = np.dot(
        matrice_embeddings,
        embedding_domanda,
    )

    top_indices = np.argsort(similarita)[-top_k:][::-1]

    risultati = []

    for indice in top_indices:

        risultati.append(
            {
                "score": float(similarita[indice]),
                "chunk": chunks[indice],
            }
        )

    return risultati


# ============================================================
# GENERAZIONE RISPOSTA
# ============================================================

def genera_risposta(domanda, risultati):

    contesto = "\n\n".join(
        risultato["chunk"]["testo"]
        for risultato in risultati
    )

    prompt = f"""
Sei un assistente AI specializzato nell'analisi di PDF.

Rispondi usando SOLO il contesto fornito.

Se la risposta non è presente nel contesto,
scrivi:
'Informazione non trovata nel documento.'

CONTESTO:
{contesto}

DOMANDA:
{domanda}

RISPOSTA:
"""

    risposta = llm.invoke(prompt)

    return risposta.content


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG PDF LOCALE CON OLLAMA")
    print("=" * 70)

    print("\n[INFO] Caricamento PDF...\n")

    documenti = carica_documenti()

    if not documenti:

        print("[ERROR] Nessun documento valido.")
        return

    print("\n[INFO] Creazione chunks...\n")

    chunks = crea_chunks(documenti)

    print(
        f"[SUCCESS] Creati "
        f"{len(chunks)} chunk"
    )

    print("\n[INFO] Generazione vector store...\n")

    matrice_embeddings = indicizza_chunks(chunks)

    print("\n[SUCCESS] Sistema RAG pronto.\n")

    while True:

        domanda = input("Domanda > ").strip()

        if domanda.lower() in ["exit", "quit"]:

            print("\n[INFO] Chiusura applicazione.")
            break

        if not domanda:
            continue

        print("\n[INFO] Ricerca semantica...\n")

        risultati = ricerca_semantica(
            domanda,
            chunks,
            matrice_embeddings,
        )

        risposta = genera_risposta(
            domanda,
            risultati,
        )

        print("\n" + "=" * 70)
        print("RISPOSTA")
        print("=" * 70)

        print(risposta)

        print("\nFONTI UTILIZZATE:\n")

        for i, risultato in enumerate(risultati, start=1):

            chunk = risultato["chunk"]

            print(
                f"[{i}] "
                f"{chunk['documento']} "
                f"(chunk {chunk['chunk_id']})"
            )

            print(
                f"Score: "
                f"{risultato['score']:.4f}"
            )

            print(
                chunk["testo"][:200]
                .replace("\n", " ")
            )

            print()


if __name__ == "__main__":
    main()
```
