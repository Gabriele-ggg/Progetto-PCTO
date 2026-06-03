"""Wrapper minimo per la pipeline di ingestione per (ri)generare trasporti.json dai PDF.
Implementa il semplice entrypoint ETL menzionato nel piano di progetto.
"""
from .services import rag_service


def run_ingestion(pdf_root=None):
    """Chiama il generatore in `rag_service` per produrre data/trasporti.json.
    Restituisce il percorso del file JSON generato.
    """
    if pdf_root is None:
        pdf_root = rag_service.PDF_SOURCE_ROOT
    print(f"Running ingestion: PDF root = {pdf_root}")
    path = rag_service.generate_transport_json(pdf_root)
    print(f"Ingestion complete: wrote {path}")
    return path


if __name__ == '__main__':
    run_ingestion()
