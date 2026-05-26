from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

# --- 1. CONFIGURAZIONE DEI COMPONENTI ---

# Inizializziamo il modello Ollama (es. llama3 o il modello che hai attivo)
llm = ChatOllama(
    model="llama3:8b",
)

# Configuriamo lo splitter di testo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,        # Aumentato leggermente per dare più senso ai chunk
    chunk_overlap=30,      # sovrapposizione
    length_function=len,
)

# --- 2. PREPARAZIONE DEL TESTO ---

testo_esempio = (
    "Il chunking è una tecnica fondamentale per preparare i dati per i database vettoriali. "
    "Permette di dividere documenti molto lunghi in frammenti più piccoli e gestibili. "
    "L'architettura dei Transformer e dei Large Language Model (LLM) trae enorme beneficio "
    "da questa pratica, poiché riduce il rumore di fondo nel testo e ottimizza i costi "
    "legati al consumo dei token nella finestra di contesto."
)

# --- 3. ESECUZIONE DELLA PIPELINE ---

# Fase A: Dividiamo il testo in chunk
chunks = text_splitter.split_text(testo_esempio)
print(f"Testo diviso con successo in {len(chunks)} chunk.\n")

# Fase B: Iteriamo sui chunk e chiediamo a Ollama di elaborarli
for i, chunk in enumerate(chunks):

    print(f"{i}:\n")
    # Prepariamo l'istruzione (prompt) per il modello inserendo il chunk
    prompt = f"Fai un riassunto brevissimo, in una sola riga, del seguente testo:\n\n{chunk}"
    
    # Invochiamo Ollama
    risposta = llm.invoke(prompt)
    
    # Mostriamo il risultato generato dall'LLM
    print(risposta.content)
    print("\n")