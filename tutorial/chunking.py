from langchain_text_splitters import RecursiveCharacterTextSplitter

testo_esempio = (
    "Il chunking è una tecnica fondamentale per preparare i dati per i database vettoriali. "
    "Permette di dividere documenti molto lunghi in frammenti più piccoli e gestibili."
)

# Configura lo splitter (lavora sul numero di caratteri, non di parole)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,        # Dimensione massima di ogni chunk (in caratteri)
    chunk_overlap=20,      # Sovrapposizione tra i chunk (in caratteri)
    length_function=len,
)

# Genera i chunk
chunks = text_splitter.split_text(testo_esempio)

# Mostra i risultati
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")