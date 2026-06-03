import streamlit as st
import requests
from datetime import datetime
from api_client import post_ask, get_telemetry_summary, upload_files

st.set_page_config(page_title='EduGuide AI', layout='wide')

# ─── Inizializzazione Session State ───────────────────────────────────────────
# FIX: tutto lo stato viene inizializzato qui, PRIMA di qualsiasi uso,
# così le funzioni helper sono già definite quando la sidebar le chiama.

if 'user' not in st.session_state:
    st.session_state.user = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_session' not in st.session_state:
    st.session_state.current_session = None


# ─── Funzioni helper (definite prima del loro utilizzo) ───────────────────────

def _create_new_session(first_question: str) -> str:
    sid = f"session_{int(datetime.now().timestamp())}"
    session = {
        'id':       sid,
        'title':    (first_question or '').strip()[:80],
        'created':  datetime.now().isoformat(),
        'messages': [],
    }
    st.session_state.history.append(session)
    st.session_state.current_session = sid
    return sid


def _append_to_current_session(role: str, text: str, latency_ms=None):
    sid = st.session_state.current_session
    if not sid:
        return
    for s in st.session_state.history:
        if s.get('id') == sid:
            s.setdefault('messages', []).append({
                'role':       role,
                'content':    text,
                'ts':         datetime.now().isoformat(),
                'latency_ms': latency_ms,
            })
            if role == 'user' and not s.get('title'):
                s['title'] = text.strip()[:80]
            break


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header('Accesso')
    if st.session_state.user is None:
        first = st.text_input('Nome (obbligatorio)')
        last  = st.text_input('Cognome (opzionale)')
        if st.button('Accedi'):
            if not first or not first.strip():
                st.error('Il nome è obbligatorio')
            else:
                name = f"{first.strip()} {last.strip()}".strip()
                st.session_state.user = name
                try:
                    requests.post(
                        'http://127.0.0.1:8000/api/telemetry',
                        json={'user': name, 'event': 'login', 'detail': 'streamlit'},
                        timeout=3,
                    )
                except Exception:
                    pass
                st.success(f'Accesso effettuato: {name}')
    else:
        st.markdown(f"**Utente:** {st.session_state.user}")
        if st.button('Esci'):
            st.session_state.user = None

    # --- Selezione modello LLM ---
    st.markdown('---')
    st.markdown('**Modello LLM**')
    try:
        list_resp     = requests.get('http://127.0.0.1:8000/api/list-models', timeout=5).json()
        model_options = list_resp.get('models', [])
        current_model = list_resp.get('current', '')
    except Exception:
        model_options = []
        current_model = ''
        st.warning('Non riesco a contattare il backend per i modelli disponibili.')

    selected_model = st.selectbox(
        'Seleziona modello',
        options=model_options or ['(nessun modello disponibile)'],
        index=0,
        key='__model_select',
    )
    if selected_model and selected_model != current_model and selected_model != '(nessun modello disponibile)':
        try:
            r = requests.post(
                'http://127.0.0.1:8000/api/set-model',
                json={'model': selected_model},
                timeout=5,
            )
            if r.status_code == 200:
                st.success(f'Modello impostato: {selected_model}')
                st.rerun()  # FIX: st.experimental_rerun() è deprecato
            else:
                st.error(f'Errore: {r.text}')
        except Exception as e:
            st.error(f'Impossibile contattare il backend: {e}')

    # --- Conversazioni salvate ---
    st.markdown('---')
    st.markdown('**Conversazioni salvate**')

    if st.session_state.history:
        opts   = [
            f"{(s.get('title') or s.get('id'))} — {s.get('created', '')[:19]}|{s.get('id')}"
            for s in st.session_state.history
        ]
        sel    = st.selectbox('Seleziona conversazione', options=opts, key='__select_session')
        if sel:
            sel_id         = sel.split('|')[-1]
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button('Apri', key=f'open_{sel_id}'):
                    for s in st.session_state.history:
                        if s.get('id') == sel_id:
                            st.session_state.current_session = sel_id
                            st.session_state.messages        = list(s.get('messages', []))
                            break
                    st.rerun()  # FIX: deprecato → st.rerun()
            with col2:
                if st.button('In cima', key=f'promote_{sel_id}'):
                    for i, s in enumerate(st.session_state.history):
                        if s.get('id') == sel_id:
                            item = st.session_state.history.pop(i)
                            st.session_state.history.insert(0, item)
                            st.session_state.current_session = sel_id
                            break
                    st.rerun()
            with col3:
                if st.button('Elimina', key=f'delete_{sel_id}'):
                    for i, s in enumerate(st.session_state.history):
                        if s.get('id') == sel_id:
                            st.session_state.history.pop(i)
                            if st.session_state.current_session == sel_id:
                                st.session_state.current_session = None
                                st.session_state.messages        = []
                            break
                    st.rerun()
    else:
        st.markdown('_Nessuna conversazione salvata_')

    if st.button('Nuova conversazione', key='new_conv'):
        st.session_state.messages = []
        _create_new_session('')
        st.rerun()

    # --- Telemetria ---
    st.markdown('---')
    st.markdown('**Telemetria**')
    telemetry_data = {}
    if st.button('Aggiorna telemetria', key='refresh_telemetry'):
        try:
            telemetry_data = get_telemetry_summary()
        except Exception as e:
            telemetry_data = {'error': str(e)}

    if telemetry_data:
        if isinstance(telemetry_data, dict) and 'summary' in telemetry_data:
            rows = [
                {
                    'user':           user,
                    'logins':         v.get('logins'),
                    'questions':      v.get('questions'),
                    'sessions':       v.get('sessions'),
                    'last_seen':      v.get('last_seen'),
                    'max_latency_ms': v.get('max_latency_ms'),
                }
                for user, v in telemetry_data.get('summary', {}).items()
            ]
            st.table(rows) if rows else st.write('Nessun dato di telemetria disponibile.')
        else:
            st.json(telemetry_data)

    # --- Carica PDF ---
    st.markdown('---')
    st.markdown('**Carica PDF**')
    category_map  = {
        'urbani':               'urbani',
        'extraurbani':          'extraurbani',
        'treni':                'treni',
        'circolari scolastiche': 'circolari_scuola',
    }
    target_label   = st.selectbox('Destinazione', options=list(category_map.keys()))
    uploaded_pdfs  = st.file_uploader(
        'Seleziona PDF (formato .pdf)', type=['pdf'], accept_multiple_files=True,
    )
    if st.button('Carica documenti', key='upload_pdfs'):
        if not uploaded_pdfs:
            st.error('Seleziona almeno un file PDF da caricare')
        else:
            try:
                resp = upload_files(uploaded_pdfs, category_map[target_label])
                if resp.get('status') == 'ok':
                    st.success(
                        f"Caricati {resp.get('saved', 0)} file nella categoria {resp.get('category')}"
                    )
                    if resp.get('errors'):
                        st.warning(f"Errori: {resp.get('errors')}")
                else:
                    st.error(f"Errore durante l'upload: {resp}")
            except Exception as e:
                st.error(f"Errore durante l'upload: {e}")


# ─── Gestione input utente ────────────────────────────────────────────────────

def handle_user_input():
    prompt = st.chat_input('Fai una domanda...')
    if prompt:
        if not st.session_state.current_session:
            _create_new_session(prompt)

        st.session_state.messages.append({'role': 'user', 'content': prompt})
        _append_to_current_session('user', prompt)

        with st.spinner('EduGuide AI sta elaborando i dati...'):
            try:
                response = post_ask(prompt)
                if 'response' in response:
                    ai_text = response['response']
                    latency = response.get('latency_ms')
                    st.session_state.messages.append({'role': 'assistant', 'content': ai_text})
                    _append_to_current_session('assistant', ai_text, latency_ms=latency)
                    with st.chat_message('assistant'):
                        st.markdown(ai_text)
                else:
                    st.error('Risposta API incompleta.')
            except Exception as e:
                st.error(f'Errore nella comunicazione con EduGuide AI: {str(e)}')


# ─── Layout principale ────────────────────────────────────────────────────────

st.title('EduGuide AI - Chat Trasporti')
for m in st.session_state.messages:
    with st.chat_message(m.get('role')):
        st.markdown(m.get('content'))

handle_user_input()