import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # STRUMENTO DI CHUNKING
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# CONFIGURAZIONE SWITCH MODELLO
# ==========================================
MODELLO_SCELTO = "nomic-embed-text:latest" 
print(f"🔄 Modello di embedding attivo: {MODELLO_SCELTO}")

MODELLO_SAFE_NAME = MODELLO_SCELTO.replace(":", "-")
PERSIST_DIR = os.path.join(".", "chroma_db", MODELLO_SAFE_NAME)
os.makedirs(PERSIST_DIR, exist_ok=True)

embeddings_model = OllamaEmbeddings(model=MODELLO_SCELTO)

# ==========================================
# CONFIGURAZIONE CHUNKING (Configura qui le grandezze)
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Lunghezza massima di ogni pezzo (in caratteri)
    chunk_overlap=70,     # Quanti caratteri sovrapporre tra un pezzo e l'altro
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

# ==========================================
# SALVATAGGIO NEL DB (VERSIONE REALE IN BATCH)
# ==========================================
import ollama  # Assicurati di averlo installato: pip install ollama

print(f"🧠 Inizializzazione del Database vettoriale...")
vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings_model
)

# Con all-minilm 200-500 è perfetto per un vero batch
BATCH_SIZE = 500 

print(f"🚀 Invio massivo a Ollama e scrittura nel DB in blocchi da {BATCH_SIZE}...")

for i in range(0, len(documenti_spezzettati), BATCH_SIZE):
    batch = documenti_spezzettati[i:i + BATCH_SIZE]
    
    # 1. Estraiamo solo i testi dal formato Document di LangChain
    testi_batch = [doc.page_content for doc in batch]
    metadati_batch = [doc.metadata for doc in batch]
    
    # 2. CHIAMATA RAPIDA: Chiediamo a Ollama tutti gli embedding del batch insieme
    # Questa singola chiamata genera 500 vettori in parallelo
    risposta_ollama = ollama.embed(
        model="all-minilm:latest",
        input=testi_batch
    )
    vettori_batch = risposta_ollama['embeddings']
    
    # Generiamo degli ID univoci per questo batch (necessari per Chroma)
    ids_batch = [f"id_{j}" for j in range(i, i + len(batch))]
    
    # 3. SCRITTURA DI MASSA: Passiamo i vettori già pronti a Chroma
    vector_db.add_vectors(
        vectors=vettori_batch,
        documents=testi_batch,
        metadatas=metadati_batch,
        ids=ids_batch
    )
    
    print(f"📊 Progressi: {i + len(batch)} / {len(documenti_spezzettati)} chunk scritti nel DB.")

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