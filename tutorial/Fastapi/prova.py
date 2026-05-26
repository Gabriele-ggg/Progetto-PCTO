from fastapi import FastAPI

# Inizializza l'applicazione
app = FastAPI()

# Crea il tuo primo "endpoint" (una rotta API)
@app.get("/")
def read_root():
    return {"messaggio": "Benvenuto in FastAPI!"}

