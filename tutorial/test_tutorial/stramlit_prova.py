import streamlit as st
import pandas as pd

# 1. Titolo dell'applicazione
st.title("La mia prima App con Streamlit!")

# 2. SEZIONE BENVENUTO (Richiesta da te)
st.subheader("Sezione Benvenuto")

# Crea una casella di testo per inserire il nome
nome = st.text_input("Inserisci il tuo nome:")

# Crea un pulsante. Il codice dentro l' 'if' si attiva solo quando lo clicchi
if st.button("Saluta"):
    if nome:  # Controlla se l'utente ha scritto qualcosa
        st.success(f"Benvenuto {nome}! 👋")
    else:
        st.warning("Per favore, inserisci un nome prima di cliccare!")

st.markdown("---") # Una linea di separazione visiva

# 3. SEZIONE GRAFICO INTERATTIVO (Esempio precedente)
st.subheader("Sezione Grafico Interattivo")

# Crea uno slider per l'utente
numero = st.slider("Seleziona un valore per modificare il grafico", 1, 100, 25)

st.write(f"Hai selezionato il numero: {numero}")

# Mostra un grafico che reagisce allo slider
dati = pd.DataFrame({
    "colonna_A": [1, 2, 3, 4], 
    "colonna_B": [10, 20, numero, 40]
})
st.line_chart(dati)


# comandi
#pip install streamlit

# entri nella cartella dove si trova il tuo file python
#
# cd...

#   streamlit run app.py  # il tuo file python

# poi copia il link