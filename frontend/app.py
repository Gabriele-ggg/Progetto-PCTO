import streamlit as st
import requests
from datetime import datetime
from api_client import post_ask

st.set_page_config(page_title="EduGuide AI", layout="wide")

# Autenticazione semplice: nome obbligatorio, cognome opzionale
if 'user' not in st.session_state:
    st.session_state.user = None
    # History per-user: lista di sessioni (id, title, created, messages[])
    st.session_state.history = st.session_state.get('history', [])
    # ID della sessione corrente
    st.session_state.current_session = None

with st.sidebar:
    st.header('Accesso')
    if st.session_state.user is None:
        first = st.text_input('Nome (obbligatorio)')
        last = st.text_input('Cognome (opzionale)')
        if st.button('Accedi'):
            if not first or not first.strip():
                st.error('Il nome è obbligatorio')
            else:
                name = f"{first.strip()} {last.strip()}".strip()
                st.session_state.user = name
                try:
                    requests.post('http://127.0.0.1:8000/api/telemetry', json={'user': name, 'event': 'login', 'detail': 'streamlit'})
                except Exception:
                    pass
                st.success(f'Accesso effettuato: {name}')
    else:
        st.markdown(f"**Utente:** {st.session_state.user}")
        if st.button('Esci'):
            st.session_state.user = None

    # --- Selezione modello LLM (dinamicamente da Ollama) ---
    st.markdown('---')
    st.markdown('**Modello LLM**')
    try:
        list_resp = requests.get('http://127.0.0.1:8000/api/list-models', timeout=5).json()
        model_options = list_resp.get('models', ['rule-based'])
        current_model = list_resp.get('current', 'rule-based')
    except Exception as e:
        model_options = ['rule-based']
        current_model = 'rule-based'
        st.warning(f'Non riesco a contattare il backend per i modelli disponibili.')

    selected_model = st.selectbox(
        'Seleziona modello',
        options=model_options,
        index=model_options.index(current_model) if current_model in model_options else 0,
        key='__model_select',
    )
    if selected_model != current_model:
        try:
            r = requests.post(
                'http://127.0.0.1:8000/api/set-model',
                json={'model': selected_model},
                timeout=5,
            )
            if r.status_code == 200:
                st.success(f'Modello impostato: {selected_model}')
                st.experimental_rerun()
            else:
                st.error(f'Errore: {r.text}')
        except Exception as e:
            st.error(f'Impossibile contattare il backend: {e}')

    # --- Gestione sessioni (crea, apri, promuovi, elimina) ---
    st.markdown('---')
    st.markdown('**Conversazioni salvate**')
    if st.session_state.history:
        # Mostra una selectbox con le sessioni e bottoni per le azioni
        opts = [f"{(s.get('title') or s.get('id'))} — {s.get('created', '')[:19]}|{s.get('id')}" for s in st.session_state.history]
        sel = st.selectbox('Seleziona conversazione', options=opts, key='__select_session')
        if sel:
            # estrai id
            sel_id = sel.split('|')[-1]
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button('Apri conversazione', key=f'open_{sel_id}'):
                    # Carica messaggi della sessione nella UI
                    for s in st.session_state.history:
                        if s.get('id') == sel_id:
                            st.session_state.current_session = sel_id
                            st.session_state.messages = [m for m in s.get('messages', [])]
                            break
                    st.experimental_rerun()
            with col2:
                if st.button('Promuovi in cima', key=f'promote_{sel_id}'):
                    for i, s in enumerate(st.session_state.history):
                        if s.get('id') == sel_id:
                            item = st.session_state.history.pop(i)
                            st.session_state.history.insert(0, item)
                            st.session_state.current_session = sel_id
                            break
                    st.experimental_rerun()
            with col3:
                if st.button('Elimina', key=f'delete_{sel_id}'):
                    for i, s in enumerate(st.session_state.history):
                        if s.get('id') == sel_id:
                            st.session_state.history.pop(i)
                            if st.session_state.current_session == sel_id:
                                st.session_state.current_session = None
                                st.session_state.messages = []
                            break
                    st.experimental_rerun()
    else:
        st.markdown('_Nessuna conversazione salvata_')

    if st.button('Nuova conversazione', key='new_conv'):
        # svuota UI e crea una nuova sessione vuota
        st.session_state.messages = []
        new_id = _create_new_session('')
        st.session_state.current_session = new_id
        st.experimental_rerun()

# Inizializzazione Session State (Memoria Chat Persistente)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "current_session" not in st.session_state:
    st.session_state.current_session = None

# Funzione per gestire la risposta dell'utente
def _create_new_session(first_question: str) -> str:
    sid = f"session_{int(datetime.now().timestamp())}"
    session = {
        "id": sid,
        "title": (first_question or "").strip()[:80],
        "created": datetime.now().isoformat(),
        "messages": []
    }
    st.session_state.history.append(session)
    st.session_state.current_session = sid
    return sid


def _append_to_current_session(role: str, text: str, latency_ms=None):
    sid = st.session_state.current_session
    if not sid:
        return
    for s in st.session_state.history:
        if s.get("id") == sid:
            s.setdefault("messages", []).append({
                "role": role,
                "content": text,
                "ts": datetime.now().isoformat(),
                "latency_ms": latency_ms,
            })
            # update session title if it's the first user message
            if role == "user" and (not s.get("title")):
                s["title"] = text.strip()[:80]
            break


def handle_user_input():
    prompt = st.chat_input("Fai una domanda...")
    if prompt:
        # Ensure there is a current session
        if not st.session_state.current_session:
            _create_new_session(prompt)

        # Save user message to UI and history
        st.session_state.messages.append({"role": "user", "content": prompt})
        _append_to_current_session("user", prompt, latency_ms=None)

        with st.spinner("EduGuide AI sta elaborando i dati..."):
            try:
                response = post_ask(prompt)
                if "response" in response:
                    ai_text = response["response"]
                    latency = response.get("latency_ms")
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    _append_to_current_session("assistant", ai_text, latency_ms=latency)
                    with st.chat_message("assistant"):
                        st.markdown(ai_text)
                else:
                    st.error("Risposta API incompleta.")
            except Exception as e:
                st.error(f"Errore nella comunicazione con EduGuide AI: {str(e)}")

# Layout principale
st.title('EduGuide AI - Chat Trasporti')
for m in st.session_state.messages:
    role = m.get('role')
    content = m.get('content')
    with st.chat_message(role):
        st.markdown(content)

handle_user_input()