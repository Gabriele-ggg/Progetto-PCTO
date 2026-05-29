from langchain_core.prompts import ChatPromptTemplate


from datetime import datetime
from zoneinfo import ZoneInfo  # Modulo integrato

# Data e ora attuale in un fuso orario specifico
roma_time = datetime.now(ZoneInfo("Europe/Rome"))

day = roma_time.strftime("%A")
hour = roma_time.strftime("%H")
mint = roma_time.strftime("%M")

tim = [day, hour, mint]
stringa_tempo = f"{tim[0]}, {tim[1]}:{tim[2]}"

# Creazione del template strutturato
assistente_autobus_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        "Tu sei l'assistente virtuale ufficiale di 'BusTravel'. Il tuo compito è aiutare gli utenti a trovare orari, prezzi e tratte dei pullman. "
        "Sii sempre cortese, preciso e rispondi *solo* con informazioni legate ai viaggi in autobus. "
        "Se l'utente ti chiede cose non pertinenti (es. voli, treni, ricette di cucina), rifiuta gentilmente di rispondere."
    ),
    (
        "human", 
        "Vorrei informazioni per un viaggio da {partenza} a {destinazione}"
    )
])

# Simulazione di inserimento dati da parte dell'utente
prompt_pronto = assistente_autobus_template.format_messages(
    partenza="Milano",
    destinazione="Roma",
    data=stringa_tempo,
)