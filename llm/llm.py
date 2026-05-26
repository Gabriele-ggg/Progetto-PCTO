import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

PATH_CARTELLA = "documenti_pdf"
MODELLO_LLM = "llama3"
MODELLO_EMBEDDINGS = "nomic-embed-text"

def inizializza_sistema():
    if not os.path.exists(PATH_CARTELLA) or not os.listdir(PATH_CARTELLA):
        print(f"❌ Errore: La cartella '{PATH_CARTELLA}' è vuota o non esiste.")
        return None

    print("📖 Caricamento dei PDF...")
    loader = PyPDFDirectoryLoader(PATH_CARTELLA)
    documenti = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    blocchi_testo = text_splitter.split_documents(documenti)
    
    print("🧠 Creazione database vettoriale...")
    embeddings = OllamaEmbeddings(model=MODELLO_EMBEDDINGS)
    vector_store = Chroma.from_documents(documents=blocchi_testo, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model=MODELLO_LLM, temperature=0.3)

    system_prompt = (
        "Sei un assistente che risponde alle domande basandosi sul contesto fornito.\n"
        "Rispondi in modo chiaro e in italiano. Se non sai la risposta, dì che non la sai.\n\n"
        "Contesto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

if __name__ == "__main__":
    catena_rag = inizializza_sistema()
    if catena_rag:
        print("\n🚀 Sistema Pronto! Fai una domanda (scrivi 'esci' per terminare):\n")
        while True:
            domanda = input("❓ Tu: ")
            if domanda.lower() in ['esci', 'exit']:
                break
            if not domanda.strip():
                continue
            risposta = catena_rag.invoke({"input": domanda})
            print(f"\n💡 Risposta:\n{risposta['answer']}\n" + "-"*50)