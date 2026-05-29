"""
chroma_store.py — Giorno 3
Indicizza documenti in ChromaDB locale e confronta i modelli di embedding
Nomic (nomic-embed-text) vs MxBai (mxbai-embed-large).

Utilizzo:
    # Indicizzare con Nomic (default)
    python chroma_store.py --action index   --model nomic

    # Indicizzare con MxBai
    python chroma_store.py --action index   --model mxbai

    # Interrogare il DB (cosine similarity)
    python chroma_store.py --action query   --model nomic --query "orari treno Gallarate Milano"

    # Confrontare i due modelli sulla stessa query
    python chroma_store.py --action compare --query "autobus Busto Arsizio mattina"

Dipendenze:
    pip install langchain langchain-community langchain-chroma chromadb ollama
"""

import argparse
import json
import time
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


# ---------------------------------------------------------------------------
# Configurazione modelli
# ---------------------------------------------------------------------------

MODELLI = {
    "nomic": "nomic-embed-text",
    "mxbai": "mxbai-embed-large"
}

CHROMA_BASE = "./chroma_db"


def get_embedding_model(nome_modello: str) -> OllamaEmbeddings:
    """Restituisce il modello di embedding Ollama selezionato."""
    model_id = MODELLI.get(nome_modello)
    if not model_id:
        raise ValueError(f"Modello '{nome_modello}' non valido. Scegli: {list(MODELLI.keys())}")
    print(f"[EMBED] Modello selezionato: {model_id}")
    return OllamaEmbeddings(model=model_id)


def get_persist_dir(nome_modello: str) -> str:
    """Directory separata per ogni modello, così si confrontano facilmente."""
    return f"{CHROMA_BASE}_{nome_modello}"


# ---------------------------------------------------------------------------
# Caricamento documenti da trasporti.json
# ---------------------------------------------------------------------------

def carica_documenti(path: str = "trasporti.json") -> list[Document]:
    """Legge trasporti.json e costruisce Document LangChain (riusa logica Giorno 2)."""
    with open(path, encoding="utf-8") as f:
        dati = json.load(f)

    documenti = []
    for linea in dati.get("linee", []):
        nome_linea = linea["nome"]
        tipo = linea["tipo"]
        for corsa in linea.get("corse", []):
            fermate_testo = "\n".join(
                f"{f['orario']} — {f['nome']}" for f in corsa["fermate"]
            )
            testo = (
                f"Linea: {nome_linea} ({tipo})\n"
                f"Corsa: {corsa['id']} | Giorni: {', '.join(corsa['giorni'])}\n"
                f"Fermate:\n{fermate_testo}"
            )
            documenti.append(Document(
                page_content=testo,
                metadata={"linea": nome_linea, "corsa": corsa["id"]}
            ))
    return documenti


# ---------------------------------------------------------------------------
# Indicizzazione
# ---------------------------------------------------------------------------

def indicizza(nome_modello: str, source_path: str = "trasporti.json"):
    """Crea o sovrascrive il Vector DB ChromaDB con il modello scelto."""
    persist_dir = get_persist_dir(nome_modello)
    embeddings = get_embedding_model(nome_modello)
    documenti = carica_documenti(source_path)

    print(f"[CHROMA] Indicizzazione {len(documenti)} documenti → {persist_dir}")
    t0 = time.time()

    # Chroma.from_documents crea la collection e calcola gli embedding
    vectorstore = Chroma.from_documents(
        documents=documenti,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=f"trasporti_{nome_modello}"
    )

    elapsed = time.time() - t0
    print(f"  → Completato in {elapsed:.2f}s")
    print(f"  → {vectorstore._collection.count()} vettori nel DB")
    return vectorstore


# ---------------------------------------------------------------------------
# Interrogazione (similarity search)
# ---------------------------------------------------------------------------

def interroga(nome_modello: str, query: str, k: int = 3):
    """Cerca i documenti più simili alla query usando cosine similarity."""
    persist_dir = get_persist_dir(nome_modello)

    if not Path(persist_dir).exists():
        print(f"[ERRORE] DB non trovato in {persist_dir}. Esegui prima --action index")
        return []

    embeddings = get_embedding_model(nome_modello)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=f"trasporti_{nome_modello}"
    )

    print(f"\n[QUERY] '{query}' (modello: {nome_modello})")
    t0 = time.time()

    # similarity_search_with_score restituisce (Document, distanza coseno)
    risultati = vectorstore.similarity_search_with_score(query, k=k)
    elapsed = time.time() - t0

    print(f"  → {len(risultati)} risultati in {elapsed*1000:.1f}ms\n")
    for i, (doc, score) in enumerate(risultati, 1):
        # Chroma restituisce distanza L2; score basso = più simile
        print(f"  [{i}] Score: {score:.4f} | Linea: {doc.metadata.get('linea')} | Corsa: {doc.metadata.get('corsa')}")
        print(f"       {doc.page_content[:120].replace(chr(10), ' ')}")
        print()

    return risultati


# ---------------------------------------------------------------------------
# Confronto tra modelli (esercizio chiave Giorno 3)
# ---------------------------------------------------------------------------

def confronta(query: str, k: int = 3):
    """
    Esegue la stessa query su entrambi i modelli e mette a confronto i risultati.
    Mostra chiaramente le differenze di ranking e similarity score.
    """
    print("=" * 60)
    print(f"CONFRONTO MODELLI — Query: '{query}'")
    print("=" * 60)

    risultati = {}
    for nome in MODELLI:
        persist_dir = get_persist_dir(nome)
        if not Path(persist_dir).exists():
            print(f"[SKIP] {nome} — DB non trovato. Esegui prima: --action index --model {nome}")
            continue

        embeddings = get_embedding_model(nome)
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=f"trasporti_{nome}"
        )
        t0 = time.time()
        res = vectorstore.similarity_search_with_score(query, k=k)
        elapsed = time.time() - t0
        risultati[nome] = {"risultati": res, "tempo_ms": elapsed * 1000}

    # Stampa tabella comparativa
    print(f"\n{'Rank':<6} {'Nomic score':<16} {'MxBai score':<16} {'Documento'}")
    print("-" * 70)
    max_k = max(len(v["risultati"]) for v in risultati.values()) if risultati else 0
    for i in range(max_k):
        nomic_info = ""
        mxbai_info = ""
        doc_label = ""
        if "nomic" in risultati and i < len(risultati["nomic"]["risultati"]):
            doc, score = risultati["nomic"]["risultati"][i]
            nomic_info = f"{score:.4f}"
            doc_label = f"{doc.metadata.get('linea')} {doc.metadata.get('corsa')}"
        if "mxbai" in risultati and i < len(risultati["mxbai"]["risultati"]):
            doc, score = risultati["mxbai"]["risultati"][i]
            mxbai_info = f"{score:.4f}"
            if not doc_label:
                doc_label = f"{doc.metadata.get('linea')} {doc.metadata.get('corsa')}"
        print(f"{i+1:<6} {nomic_info:<16} {mxbai_info:<16} {doc_label}")

    if risultati:
        print("\nTempi di risposta:")
        for nome, dati in risultati.items():
            print(f"  {nome:<10}: {dati['tempo_ms']:.1f}ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChromaDB — Giorno 3")
    parser.add_argument("--action", choices=["index", "query", "compare"], required=True)
    parser.add_argument("--model", choices=["nomic", "mxbai"], default="nomic",
                        help="Modello embedding (per index e query)")
    parser.add_argument("--query", type=str, default="treno per Milano mattina",
                        help="Testo della query (per query e compare)")
    parser.add_argument("--source", type=str, default="trasporti.json",
                        help="File JSON sorgente (per index)")
    parser.add_argument("--k", type=int, default=3, help="Numero di risultati")
    args = parser.parse_args()

    if args.action == "index":
        indicizza(args.model, args.source)
    elif args.action == "query":
        interroga(args.model, args.query, args.k)
    elif args.action == "compare":
        confronta(args.query, args.k)


if __name__ == "__main__":
    main()