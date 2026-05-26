from langchain_ollama import ChatOllama

testo = input("Inserisci una domanda: ")

# 1. Inizializza il modello Ollama (assicurati che il nome corrisponda a quello scaricato)
llm = ChatOllama(
    model="llama3:8b",
    temperature=0.7,
    # Puoi aggiungere altri parametri come num_predict, top_p, ecc.
)

# 2. Invia un messaggio al modello
response = llm.invoke(testo)

# 3. Stampa la risposta del modello
print(response.content)