import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

# --- 1. CONFIGURAZIONE DEI COMPONENTI ---

# Numero massimo di chunk da elaborare (None = tutti)
MAX_CHUNKS = 5

llm = ChatOllama(
    model="llama3:8b",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len,
)

# --- 2. CARICAMENTO DEL TESTO DA FILE JSON ---

def carica_testo_da_json(percorso_file: str) -> str:
    path = Path("D:\prova_git\Progetto-PCTO\orari_output.json")
    
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {percorso_file}")
    
    with open(path, "r", encoding="utf-8") as f:
        dati = json.load(f)
    
    if isinstance(dati, str):
        return dati
    
    elif isinstance(dati, dict):
        for chiave in ("testo", "text", "contenuto", "content", "body"):
            if chiave in dati:
                return dati[chiave]
        return " ".join(str(v) for v in dati.values() if isinstance(v, str))
    
    elif isinstance(dati, list):
        frammenti = []
        for elemento in dati:
            if isinstance(elemento, str):
                frammenti.append(elemento)
            elif isinstance(elemento, dict):
                for chiave in ("testo", "text", "contenuto", "content", "body"):
                    if chiave in elemento:
                        frammenti.append(elemento[chiave])
                        break
        return " ".join(frammenti)
    
    else:
        raise ValueError(f"Struttura JSON non supportata: {type(dati)}")


PERCORSO_JSON = "dati.json"

testo_esempio = carica_testo_da_json(PERCORSO_JSON)
print(f"Testo caricato dal file '{PERCORSO_JSON}' ({len(testo_esempio)} caratteri).\n")

# --- 3. ESECUZIONE DELLA PIPELINE ---

chunks = text_splitter.split_text(testo_esempio)
print(f"Testo diviso in {len(chunks)} chunk totali.\n")

# Selezioniamo i chunk da elaborare in base a MAX_CHUNKS
chunks_da_elaborare = chunks if MAX_CHUNKS is None else chunks[:MAX_CHUNKS]
print(f"Chunk da elaborare: {len(chunks_da_elaborare)}"
      + (" (tutti)" if MAX_CHUNKS is None else f" (limite impostato: {MAX_CHUNKS})") + "\n")

for i, chunk in enumerate(chunks_da_elaborare):
    print(f"{i}:\n")
    prompt = f"Fai un riassunto brevissimo, in una sola riga, del seguente testo:\n\n{chunk}"
    risposta = llm.invoke(prompt)
    print(risposta.content)
    print("\n")