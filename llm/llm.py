from langchain_ollama import ChatOllama

def interroga_con_langchain():
    # Prendi la domanda dall'utente
    domanda = input("Inserisci la tua domanda (tramite LangChain): ")
    
    print("\n[LangChain sta invocando Ollama...]\n")
    
    try:
        # Inizializziamo il modello tramite LangChain
        # Puoi anche regolare la 'temperature' (es. 0 per risposte precise, 0.7 per più creative)
        llm = ChatOllama(
            model="llama3:8b",
            temperature=0.7
        )
        
        # Inviamo la domanda al modello usando il metodo standard .invoke()
        risposta = llm.invoke(domanda)
        
        # LangChain restituisce un oggetto AIMessage. 
        # Il testo pulito si trova dentro l'attributo .content
        print("====== RISPOSTA AI (LANGCHAIN) ======")
        print(risposta.content)
        print("=====================================")
        
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        print("Assicurati che Ollama sia attivo e che il modello sia corretto.")

if __name__ == "__main__":
    interroga_con_langchain()