import streamlit as st
import requests
from datetime import datetime
from api_client import post_ask, get_telemetry_summary, get_telemetry_metrics, upload_files, post_feedback

st.set_page_config(page_title='EduGuide AI', layout='wide')

# ─── Inizializzazione Session State ───────────────────────────────────────────
if 'user' not in st.session_state:
    st.session_state.user = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'show_telemetry' not in st.session_state:
    st.session_state.show_telemetry = False
if 'telemetry_data' not in st.session_state:
    st.session_state.telemetry_data = None


# ─── Funzioni helper ──────────────────────────────────────────────────────────

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


def _delete_current_chat():
    st.session_state.messages = []
    sid = st.session_state.current_session
    if sid:
        st.session_state.history = [s for s in st.session_state.history if s.get('id') != sid]
    st.session_state.current_session = None


def _fmt_ms(v):
    """Formatta un valore ms come stringa leggibile."""
    if v is None:
        return '—'
    try:
        return f"{float(v):.0f} ms"
    except Exception:
        return '—'


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
                st.rerun()
            else:
                st.error(f'Errore: {r.text}')
        except Exception as e:
            st.error(f'Impossibile contattare il backend: {e}')

    # --- Conversazioni salvate ---
    st.markdown('---')
    st.markdown('**Conversazioni salvate**')

    if st.session_state.history:
        opts = [
            f"{(s.get('title') or s.get('id'))} — {s.get('created', '')[:19]}|{s.get('id')}"
            for s in st.session_state.history
        ]
        sel = st.selectbox('Seleziona conversazione', options=opts, key='__select_session')
        if sel:
            sel_id = sel.split('|')[-1]
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button('Apri', key=f'open_{sel_id}'):
                    for s in st.session_state.history:
                        if s.get('id') == sel_id:
                            st.session_state.current_session = sel_id
                            st.session_state.messages = list(s.get('messages', []))
                            break
                    st.rerun()
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
                                st.session_state.messages = []
                            break
                    st.rerun()
    else:
        st.markdown('_Nessuna conversazione salvata_')

    st.markdown('---')
    if st.button('✨ Nuova conversazione', key='new_conv', use_container_width=True, type='primary'):
        st.session_state.messages = []
        _create_new_session('')
        st.rerun()

    # --- Elimina chat corrente ---
    st.markdown('---')
    _, col_del = st.columns([1, 2])
    with col_del:
        if st.button('🗑️ Elimina chat', key='delete_current_chat', type='secondary'):
            _delete_current_chat()
            st.rerun()

    # --- Telemetria ---
    st.markdown('---')
    st.markdown('**Telemetria**')
    if st.button('📊 Apri pannello metriche', key='open_telemetry'):
        st.session_state.show_telemetry = True
        st.session_state.telemetry_data = get_telemetry_metrics()

    # --- Carica PDF ---
    st.markdown('---')
    st.markdown('**Carica PDF**')
    category_map = {
        'urbani':                'urbani',
        'extraurbani':           'extraurbani',
        'treni':                 'treni',
        'circolari scolastiche': 'circolari_scuola',
    }
    target_label  = st.selectbox('Destinazione', options=list(category_map.keys()))
    uploaded_pdfs = st.file_uploader(
        'Seleziona PDF (formato .pdf)', type=['pdf'], accept_multiple_files=True,
    )
    if st.button('Carica documenti', key='upload_pdfs'):
        if not uploaded_pdfs:
            st.error('Seleziona almeno un file PDF da caricare')
        else:
            try:
                resp = upload_files(uploaded_pdfs, category_map[target_label])
                if resp.get('status') == 'ok':
                    st.success(f"Caricati {resp.get('saved', 0)} file nella categoria {resp.get('category')}")
                    if resp.get('errors'):
                        st.warning(f"Errori: {resp.get('errors')}")
                else:
                    st.error(f"Errore durante l'upload: {resp}")
            except Exception as e:
                st.error(f"Errore durante l'upload: {e}")


# ─── Overlay Telemetria ────────────────────────────────────────────────────────

if st.session_state.show_telemetry:
    with st.container():
        st.markdown('---')
        header_col, close_col = st.columns([8, 1])
        with header_col:
            st.subheader('📊 Telemetria in tempo reale')
        with close_col:
            if st.button('✕ Chiudi', key='close_telemetry'):
                st.session_state.show_telemetry = False
                st.rerun()

        td = st.session_state.telemetry_data or {}

        if 'error' in td:
            st.error(f"Errore: {td['error']}")
        else:
            overall  = td.get('overall', {})
            models   = td.get('models', [])
            by_user  = td.get('by_user', {})

            active_model = models[-1] if models else '—'
            min_ms  = overall.get('min_latency_ms')
            avg_ms  = overall.get('avg_latency_ms')
            max_ms  = overall.get('max_latency_ms')
            count   = overall.get('count', 0)

            # 4 colonne principali
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('🤖 Modello AI', active_model, f"{len(models)} modell{'o' if len(models)==1 else 'i'}")
            c2.metric('⚡ Tempo Min', _fmt_ms(min_ms), 'risposta più rapida')
            c3.metric('📊 Tempo Medio', _fmt_ms(avg_ms), f'{count} richieste totali')
            c4.metric('🐢 Tempo Max', _fmt_ms(max_ms), 'risposta più lenta')

            # Tabella per utente
            if by_user:
                st.markdown('#### Dettaglio per utente')
                rows = []
                for user_name, info in by_user.items():
                    rows.append({
                        'Utente':    user_name,
                        'Modello':   ', '.join(info.get('models', [])) or '—',
                        'Min (ms)':  _fmt_ms(info.get('min_latency_ms')),
                        'Media (ms)': _fmt_ms(info.get('avg_latency_ms')),
                        'Max (ms)':  _fmt_ms(info.get('max_latency_ms')),
                        'Richieste': info.get('count', 0),
                    })
                st.table(rows)
            else:
                st.info('Nessun dato per utente disponibile.')

        if st.button('↻ Aggiorna metriche', key='refresh_metrics'):
            st.session_state.telemetry_data = get_telemetry_metrics()
            st.rerun()

        st.markdown('---')


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
                        # Feedback buttons inline
                        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
                        with fb_col1:
                            if st.button('👍', key=f'up_{id(ai_text)}', help='Risposta utile'):
                                post_feedback(
                                    st.session_state.user or 'anonimo',
                                    st.session_state.current_session or '',
                                    'assistant', ai_text, 'positive'
                                )
                                st.toast('Grazie per il feedback positivo!', icon='👍')
                        with fb_col2:
                            if st.button('👎', key=f'dn_{id(ai_text)}', help='Risposta non utile'):
                                post_feedback(
                                    st.session_state.user or 'anonimo',
                                    st.session_state.current_session or '',
                                    'assistant', ai_text, 'negative'
                                )
                                st.toast('Grazie per il feedback!', icon='👎')
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