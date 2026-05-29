import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # STRUMENTO DI CHUNKING
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# CONFIGURAZIONE SWITCH MODELLO
# ==========================================
MODELLO_SCELTO = "qwen3-embedding:8b" 
print(f"🔄 Modello di embedding attivo: {MODELLO_SCELTO}")

MODELLO_SAFE_NAME = MODELLO_SCELTO.replace(":", "-")
PERSIST_DIR = os.path.join(".", "chroma_db", MODELLO_SAFE_NAME)
os.makedirs(PERSIST_DIR, exist_ok=True)

embeddings_model = OllamaEmbeddings(model=MODELLO_SCELTO)

# ==========================================
# CONFIGURAZIONE CHUNKING (Configura qui le grandezze)
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20000,       # Lunghezza massima di ogni pezzo (in caratteri)
    chunk_overlap=100,     # Quanti caratteri sovrapporre tra un pezzo e l'altro
    length_function=len,
    separators=["\n\n", "\n", " ", ""] # Prova a tagliare prima per paragrafi, poi frasi, poi parole
)

# ==========================================
# GESTIONE E CARICAMENTO PDF
# ==========================================
pagine_grezze = []

base_dir = os.path.dirname(os.path.abspath(__file__))
cartelle_pdf = {
    "urbani": os.path.join(base_dir, "pdf", "urbani"),
    "extraurbani": os.path.join(base_dir, "pdf", "extraurbani")
}

# Esplorazione e lettura dei file
for categoria, percorso_cartella in cartelle_pdf.items():
    if not os.path.exists(percorso_cartella):
        os.makedirs(percorso_cartella, exist_ok=True)
        print(f"📁 Creata cartella vuota: {percorso_cartella}")
        continue

    print(f"\n📂 Esplorazione cartella [{categoria}]...")
    file_nella_cartella = os.listdir(percorso_cartella)
    
    for nome_file in file_nella_cartella:
        if nome_file.lower().endswith('.pdf'):
            percorso_completo_pdf = os.path.join(percorso_cartella, nome_file)
            print(f"📄 Lettura: {nome_file}")
            
            try:
                reader = PdfReader(percorso_completo_pdf)
                for num_pagina, pagina in enumerate(reader.pages):
                    testo_estratto = pagina.extract_text()
                    
                    if not testo_estratto or not testo_estratto.strip():
                        continue
                    
                    # Creiamo un documento temporaneo per la pagina intera
                    doc_pagina = Document(
                        page_content=testo_estratto,
                        metadata={
                            "categoria": categoria,
                            "source": nome_file,
                            "page": num_pagina + 1
                        }
                    )
                    pagine_grezze.append(doc_pagina)
            except Exception as e:
                print(f"❌ Errore durante la lettura di {nome_file}: {e}")

if not pagine_grezze:
    print("\n⚠️ Nessun testo da elaborare. Inserisci i PDF e riprova.")
    exit()

# ==========================================
# APPLICAZIONE DEL CHUNKING
# ==========================================
print(f"\n✂️ Applicazione chunking su {len(pagine_grezze)} pagine grezze...")

# split_documents prende le pagine intere e le divide in frammenti più piccoli,
# ereditando e copiando in automatico i metadati corretti (categoria, source, page) per ogni frammento!
documenti_spezzettati = text_splitter.split_documents(pagine_grezze)

print(f"🧩 Testo suddiviso con successo in {len(documenti_spezzettati)} frammenti (chunk).")

# ==========================================
# SALVATAGGIO NEL DB
# ==========================================
print(f"🧠 Invio dei chunk a Ollama e salvataggio nel DB permanente...")

vector_db = Chroma.from_documents(
    documents=documenti_spezzettati, # Passiamo i pezzi piccoli, ora l'embedding ce la farà!
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR
)

print(f"✅ Database vettoriale aggiornato con successo in: {PERSIST_DIR}")


# ricerca semantica per i primi K chunks più rilevanti

# 1. Definisci la query di testo che l'utente sta cercando
query = "come arrivo a Udine stazione da Castions?"
k = 3  # Il numero di chunk più rilevanti che vuoi ottenere

# 2. Esegui la ricerca semantica con il punteggio di rilevanza
risultati = vector_db.similarity_search_with_relevance_scores(query, k=k)

# 3. Cicla sui risultati (ogni elemento è una tupla: (Documento, Punteggio))
for i, (doc, score) in enumerate(risultati, 1):
    print(f"--- Chunk #{i} (Score di Similarità: {score:.4f}) ---")
    print(doc.page_content)
    print(f"Metadati: {doc.metadata}\n")