"""
ingestion_pipeline.py — Giorno 2
Carica PDF o JSON strutturato (orari trasporti), applica chunking gerarchico
e prepara i documenti per l'indicizzazione vettoriale.

Utilizzo:
    python ingestion_pipeline.py --source pdf    --path documento.pdf   --chunk_size 1000
    python ingestion_pipeline.py --source json   --path trasporti.json  --chunk_size 500
    python ingestion_pipeline.py --source json   --path trasporti.json  --chunk_size 2000

Dipendenze:
    pip install langchain langchain-community pypdf
"""

import json
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# ---------------------------------------------------------------------------
# Caricamento sorgenti
# ---------------------------------------------------------------------------

def carica_pdf(path: str) -> list[Document]:
    """Carica un PDF pagina per pagina tramite PyPDFLoader."""
    print(f"[PDF] Caricamento: {path}")
    loader = PyPDFLoader(path)
    documenti = loader.load()
    print(f"  → {len(documenti)} pagine caricate")
    return documenti


def carica_json_trasporti(path: str) -> list[Document]:
    """
    Converte un JSON strutturato di orari trasporti in Document LangChain.
    Struttura attesa: vedi trasporti.json
    Ogni corsa diventa un Document separato con metadata (linea, fermata, ecc.)
    """
    print(f"[JSON] Caricamento: {path}")
    with open(path, encoding="utf-8") as f:
        dati = json.load(f)

    documenti = []
    for linea in dati.get("linee", []):
        nome_linea = linea.get("nome", "N/D")
        tipo = linea.get("tipo", "N/D")

        for corsa in linea.get("corse", []):
            id_corsa = corsa.get("id", "N/D")
            giorni = ", ".join(corsa.get("giorni", []))

            fermate_testo = []
            for fermata in corsa.get("fermate", []):
                fermate_testo.append(
                    f"{fermata['orario']} — {fermata['nome']}"
                )

            testo = (
                f"Linea: {nome_linea} ({tipo})\n"
                f"Corsa: {id_corsa} | Giorni: {giorni}\n"
                f"Fermate:\n" + "\n".join(fermate_testo)
            )

            doc = Document(
                page_content=testo,
                metadata={
                    "linea": nome_linea,
                    "tipo": tipo,
                    "corsa_id": id_corsa,
                    "giorni": giorni,
                    "source": path
                }
            )
            documenti.append(doc)

    print(f"  → {len(documenti)} corse estratte come documenti")
    return documenti


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def applica_chunking(
    documenti: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> list[Document]:
    """
    Suddivide i documenti in chunk usando RecursiveCharacterTextSplitter.
    I separatori gerarchici provano prima \n\n, poi \n, poi spazio, poi carattere.
    """
    print(f"\n[CHUNKING] chunk_size={chunk_size}, overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # gerarchia di split
    )
    chunks = splitter.split_documents(documenti)
    print(f"  → {len(documenti)} documenti → {len(chunks)} chunk")
    _stampa_statistiche(chunks)
    return chunks


def _stampa_statistiche(chunks: list[Document]):
    lunghezze = [len(c.page_content) for c in chunks]
    print(f"  → Lunghezza media: {sum(lunghezze) / len(lunghezze):.0f} caratteri")
    print(f"  → Min: {min(lunghezze)} | Max: {max(lunghezze)}")


# ---------------------------------------------------------------------------
# Controllo robots.txt (esercizio Giorno 2)
# ---------------------------------------------------------------------------

def controlla_robots(url_base: str, percorso: str = "/") -> bool:
    """
    Controlla se il percorso è permesso dal robots.txt del sito.
    Nota: per scraping reale usare urllib.robotparser.
    Qui è una demo educativa per capire la struttura.
    """
    import urllib.robotparser
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{url_base.rstrip('/')}/robots.txt")
    try:
        rp.read()
        permesso = rp.can_fetch("*", f"{url_base}{percorso}")
        stato = "PERMESSO" if permesso else "VIETATO"
        print(f"[ROBOTS] {url_base}{percorso} → {stato}")
        return permesso
    except Exception as e:
        print(f"[ROBOTS] Errore lettura robots.txt: {e}")
        return False


# ---------------------------------------------------------------------------
# Salvataggio risultati (debug)
# ---------------------------------------------------------------------------

def salva_chunks_json(chunks: list[Document], output_path: str = "chunks_output.json"):
    """Salva i chunk in un file JSON per ispezione manuale."""
    dati = [
        {
            "chunk_id": i,
            "lunghezza": len(c.page_content),
            "metadata": c.metadata,
            "testo": c.page_content[:300] + ("..." if len(c.page_content) > 300 else "")
        }
        for i, c in enumerate(chunks)
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dati, f, ensure_ascii=False, indent=2)
    print(f"\n[OUTPUT] Chunk salvati in: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline di ingestion documenti")
    parser.add_argument("--source", choices=["pdf", "json"], required=True)
    parser.add_argument("--path", required=True, help="Percorso del file")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--output", default="chunks_output.json")
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"Errore: file non trovato → {args.path}")
        return

    # 1. Caricamento
    if args.source == "pdf":
        documenti = carica_pdf(args.path)
    else:
        documenti = carica_json_trasporti(args.path)

    # 2. Chunking
    chunks = applica_chunking(documenti, args.chunk_size, args.chunk_overlap)

    # 3. Salvataggio debug
    salva_chunks_json(chunks, args.output)

    print("\nPipeline completata. I chunk sono pronti per l'embedding.")
    return chunks  # usabile anche come modulo importato


if __name__ == "__main__":
    main()