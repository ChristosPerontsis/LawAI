import streamlit as st
import os
import tempfile
import datetime
import re
import json
import time
import hashlib
import pandas as pd
# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document 
from pinecone import Pinecone
from sqlalchemy import create_engine, text
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Νομικός Σύμβουλος", layout="wide", page_icon="⚖️")
index_name = "nomikos-index"
USER_DB_FILE = "user_db.json"
SESSION_FILE = "active_sessions.json"

# --- 0. MODERN UI ENGINE (CSS) ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); font-family: 'Inter', sans-serif; }
        h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #1e3a8a !important; font-weight: 700; }
        [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #334155; }
        [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f8fafc !important; }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #94a3b8 !important; }
        div.stButton > button { background: #1e3a8a; color: white !important; border-radius: 8px; border: none; font-weight: 600; }
        div.stButton > button:hover { background: #172554; }
        .stTextInput input { border: 1px solid #cbd5e1; border-radius: 6px; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #ffffff; padding: 10px; border-radius: 10px; }
        .stTabs [aria-selected="true"] { background-color: #eff6ff !important; color: #1e3a8a !important; }
        /* Placeholder styling */
        ::placeholder { color: #a0aec0 !important; opacity: 1 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATABASE & AUTH FUNCTIONS ---
def get_db_connection():
    if "DATABASE_URL" in st.secrets:
        try:
            return create_engine(st.secrets["DATABASE_URL"])
        except: return None
    return None

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_user_data(username):
    # 1. Hardcoded Admin
    if username == "admin":
        return {"pass": hash_password("admin"), "firm_id": "ADMIN_Δημόσια_Βιβλιοθήκη", "role": "admin"}
    
    # 2. Try DB
    engine = get_db_connection()
    if engine:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM users WHERE username = :u"), {"u": username}).fetchone()
                if result:
                    return {"pass": result[1], "firm_id": result[2], "role": result[3]}
        except: pass
    
    # 3. Local JSON Fallback
    if not os.path.exists(USER_DB_FILE):
        return None
    try:
        with open(USER_DB_FILE, 'r') as f:
            users = json.load(f)
            return users.get(username)
    except: return None

def create_user(username, password, firm_name):
    hashed_pw = hash_password(password)
    engine = get_db_connection()
    
    if engine:
        try:
            with engine.connect() as conn:
                exists = conn.execute(text("SELECT 1 FROM users WHERE username = :u"), {"u": username}).fetchone()
                if exists: return False
                conn.execute(
                    text("INSERT INTO users (username, password_hash, firm_name, role) VALUES (:u, :p, :f, 'user')"),
                    {"u": username, "p": hashed_pw, "f": firm_name}
                )
                conn.commit()
                return True
        except: pass
            
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f: users = json.load(f)
    else: users = {}
    
    if username in users: return False
    users[username] = {"pass": hashed_pw, "firm_id": firm_name, "role": "user"}
    with open(USER_DB_FILE, 'w') as f: json.dump(users, f)
    return True

def load_sessions():
    if not os.path.exists(SESSION_FILE): return {}
    try:
        with open(SESSION_FILE, 'r') as f: return json.load(f)
    except: return {}

def save_session(username, timestamp):
    sessions = load_sessions()
    sessions[username] = timestamp
    with open(SESSION_FILE, 'w') as f: json.dump(sessions, f)

def clear_session(username):
    sessions = load_sessions()
    if username in sessions:
        del sessions[username]
        with open(SESSION_FILE, 'w') as f: json.dump(sessions, f)

def auto_genitive(name):
    if not name: return ""
    COMMON_NAMES_DB = {"ΧΡΗΣΤΟΣ": "Χρήστου", "ΠΕΡΟΝΤΣΗΣ": "Περόντση", "ΜΑΡΙΑ": "Μαρίας"}
    parts = name.split()
    gen_parts = []
    article = "ΤΟΥ" 
    if parts[0].endswith(('α', 'η', 'ω', 'Α', 'Η', 'Ω')): article = "ΤΗΣ"
    for w in parts:
        w_upper = w.upper()
        if w_upper in COMMON_NAMES_DB: gen_parts.append(COMMON_NAMES_DB[w_upper])
        elif w_upper.endswith('ΟΣ'): gen_parts.append(w[:-2] + 'ου')
        elif w_upper.endswith('ΗΣ'): gen_parts.append(w[:-2] + 'η')
        elif w_upper.endswith('ΑΣ'): gen_parts.append(w[:-1])
        elif w_upper.endswith(('Α', 'Η', 'Ω')): gen_parts.append(w + 'ς')
        else: gen_parts.append(w)
    return f"{article} {' '.join(gen_parts)}"

@st.dialog("Προσχέδιο Email")
def show_email_draft(case_name, case_email, case_debt, case_deadline, firm_name):
    st.markdown("### Επίσημη Ειδοποίηση")
    email_body = f"""Αξιότιμε/η {case_name},\n\nΣε συνέχεια της Εξώδικης Δήλωσης, η προθεσμία των 15 ημερών για την οφειλή {case_debt}€ λήγει στις {case_deadline}.\n\nΟ Πληρεξούσιος Δικηγόρος\n{firm_name}"""
    st.text_area("Κείμενο προς Αντιγραφή:", value=email_body, height=250)

# --- 3. LOGIN PAGE ---
def login_page():
    local_css()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<h1 style='text-align: center;'>⚖️ Νομικός Cloud</h1>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Σύνδεση", "Εγγραφή"])
            with tab1:
                with st.form("login"):
                    u = st.text_input("Username", key="login_u")
                    p = st.text_input("Password", type="password", key="login_p")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("ΕΙΣΟΔΟΣ", use_container_width=True):
                        with st.spinner("Γίνεται ασφαλής σύνδεση..."):
                            time.sleep(1.5) 
                            user_data = load_user_data(u)
                            if user_data and user_data["pass"] == hash_password(p):
                                new_ts = time.time()
                                save_session(u, new_ts)
                                st.session_state['logged_in'] = True
                                st.session_state['username'] = u
                                st.session_state['firm_id'] = user_data["firm_id"]
                                st.session_state['login_ts'] = new_ts
                                st.rerun()
                            else:
                                st.error("Λάθος στοιχεία.")
            with tab2:
                with st.form("signup"):
                    new_u = st.text_input("Νέο Username", key="signup_u")
                    new_p = st.text_input("Νέος Κωδικός", type="password", key="signup_p")
                    firm = st.text_input("Όνομα Γραφείου", key="signup_firm")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("ΔΗΜΙΟΥΡΓΙΑ ΛΟΓΑΡΙΑΣΜΟΥ", use_container_width=True):
                        with st.spinner("Δημιουργία λογαριασμού..."):
                            time.sleep(1)
                            if create_user(new_u, new_p, firm):
                                st.success("Επιτυχής Εγγραφή! Τώρα μπορείτε να συνδεθείτε.")
                            else:
                                st.error("Το Username υπάρχει ήδη.")

# --- 4. MAIN APPLICATION ---
def main_app():
    local_css()
    current_firm = st.session_state['firm_id']
    current_user = st.session_state['username']
    
    active_sessions = load_sessions()
    my_ts = st.session_state.get('login_ts', 0)
    server_ts = active_sessions.get(current_user, 0)
    if server_ts != my_ts:
        st.warning("⚠️ Αποσύνδεση: Συνδεθήκατε από άλλη συσκευή.")
        st.session_state['logged_in'] = False
        time.sleep(2)
        st.rerun()

    if "active_evictions" not in st.session_state: st.session_state.active_evictions = []
    if "messages" not in st.session_state: st.session_state.messages = []
    if "analysis_text" not in st.session_state: st.session_state.analysis_text = ""
    
    with st.sidebar:
        st.markdown(f"### {current_firm}")
        if st.button("🚪 Αποσύνδεση", use_container_width=True):
            clear_session(current_user)
            st.session_state['logged_in'] = False
            st.rerun()
        st.divider()
        if "ADMIN" in current_firm:
            if st.button("Διαγραφή ΟΛΩΝ (Admin)", type="primary", use_container_width=True):
                try:
                    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                    pc.Index(index_name).delete(delete_all=True)
                    st.toast("Βάση Καθαρίστηκε")
                except: st.error("Error")

    # --- AUTO DISCOVERY BRAIN ---
    if "nomikos_llm" not in st.session_state:
        try:
            with st.spinner("Finding available AI models..."):
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                all_models = list(genai.list_models())
                available_model_names = [m.name.replace("models/", "") for m in all_models if 'generateContent' in m.supported_generation_methods]
                
                selected_model = None
                for m in available_model_names:
                    if "gemini-1.5-flash" in m: selected_model = m; break
                if not selected_model:
                    for m in available_model_names:
                        if "gemini-1.5-pro" in m: selected_model = m; break
                if not selected_model and available_model_names: selected_model = available_model_names[0]
                
                if selected_model:
                    st.session_state.nomikos_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3, google_api_key=st.secrets["GOOGLE_API_KEY"])
                else:
                    st.error("No Gemini models found.")
                    st.stop()
        except Exception as e:
            st.error(f"Failed to connect to Google AI: {e}")
            st.stop()

    llm = st.session_state.nomikos_llm
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})
    except Exception as e: st.error(f"Embeddings Error: {e}"); st.stop()

    st.title("🗂️ Νομικός Φάκελος")
    
    t1, t2, t3, t4, t5, t6 = st.tabs(["Αρχειοθέτηση", "X-Ray Debugger", "Εργαλεία", "Νομικός Βοηθός", "Αυτόματη Σύνταξη", "Διαχείριση Εξώσεων"])
    
    with t1:
        st.header("Εισαγωγή Νέων Εγγράφων")
        if "ADMIN" in current_firm:
            st.info("🔓 **ADMIN MODE**: Τα αρχεία που ανεβάζετε εδώ θα είναι ορατά σε ΟΛΟΥΣ τους χρήστες (Public Library).")
        
        with st.container():
            files = st.file_uploader("Επιλέξτε αρχεία (PDF ή JSON)", type=["pdf", "json"], accept_multiple_files=True, key="uploader")
            if st.button("🔒 Αποθήκευση στη Βάση", key="btn_upload") and files:
                with st.spinner("Επεξεργασία & Καταχώρηση..."):
                    for f in files:
                        try:
                            clean_name = f.name
                            upload_type = "public" if "ADMIN" in current_firm else "private"
                            target_id = "Public_Legal_Library" if "ADMIN" in current_firm else current_firm
                            try:
                                pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                                pc.Index(index_name).delete(filter={"firm_id": target_id, "file_name": clean_name})
                            except: pass

                            if f.name.endswith(".json"):
                                try:
                                    raw_text = f.read().decode("utf-8").strip()
                                    if raw_text.startswith("{") and not raw_text.startswith("["): raw_text = f"[{raw_text}]"
                                    data = json.loads(raw_text)
                                    if isinstance(data, dict): data = [data]
                                    docs_to_upload = []
                                    for entry in data:
                                        d = Document(page_content=entry["text"], metadata={"firm_id": target_id, "source_type": upload_type, "file_name": clean_name, "article_id": entry["id"]})
                                        docs_to_upload.append(d)
                                    if docs_to_upload:
                                        PineconeVectorStore.from_documents(docs_to_upload, embeddings, index_name=index_name)
                                        st.success(f"✅ JSON '{clean_name}' uploaded successfully.")
                                except Exception as e: st.error(f"JSON Error: {e}")
                            elif f.name.endswith(".pdf"):
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    tmp.write(f.read())
                                    path = tmp.name
                                loader = PyPDFLoader(path); docs = loader.load()
                                full_text = "\n".join([page.page_content for page in docs])
                                docs_to_upload = []
                                pattern = r'(Άρθρο\s*:?\s*\d+)' 
                                parts = re.split(pattern, full_text)
                                for i in range(1, len(parts), 2):
                                    if i + 1 < len(parts):
                                        title = parts[i].strip()
                                        body = parts[i+1].strip()
                                        body = re.sub(r'\s+', ' ', body).strip()
                                        full_entry = title + "\n" + body
                                        match = re.search(r'\d+', title)
                                        art_id = match.group() if match else "0"
                                        d = Document(page_content=full_entry, metadata={"firm_id": target_id, "source_type": upload_type, "file_name": clean_name, "article_id": art_id})
                                        docs_to_upload.append(d)
                                if docs_to_upload:
                                    PineconeVectorStore.from_documents(docs_to_upload, embeddings, index_name=index_name)
                                    st.success(f"✅ PDF '{clean_name}' uploaded successfully.")
                                os.unlink(path)
                        except Exception as e: st.error(f"Error processing {f.name}: {e}")

    with t2:
        st.header("X-Ray Database Debugger")
        col1, col2 = st.columns([3, 1])
        q = col1.text_input("Αναζήτηση (π.χ. 'Άρθρο 125')", key="file_search_input")
        if st.checkbox("🔍 Debug Mode"):
            if col2.button("Αναζήτηση (Global)", key="btn_debug_search"):
                vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                res = vs.similarity_search(q, k=10) 
                if not res: st.warning("Database is empty.")
                for i, d in enumerate(res):
                    st.markdown(f"**Result {i+1}:** `{d.metadata.get('file_name')}` | Firm: `{d.metadata.get('firm_id')}` | Art: `{d.metadata.get('article_id')}`")
                    st.text(d.page_content[:200] + "...")
        else:
            if col2.button("Αναζήτηση", key="btn_file_search"):
                vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                target_ids = [current_firm, "Public_Legal_Library"]
                res = vs.similarity_search(q, k=10, filter={"firm_id": {"$in": target_ids}})
                if not res: st.warning("Δεν βρέθηκαν αποτελέσματα.")
                for i, d in enumerate(res):
                    with st.expander(f"{i+1}. {d.metadata.get('file_name')} ({d.metadata.get('firm_id')})"):
                        st.text(d.page_content)

    with t3:
        st.header("Νομικά Εργαλεία")
        tc = st.radio("Επιλέξτε λειτουργία:", ["Μετάφραση", "Ανωνυμοποίηση (GDPR)"], horizontal=True, key="tool_select")
        if tc == "Μετάφραση":
            txt = st.text_area("Κείμενο προς μετάφραση:", key="trans_input")
            lang = st.selectbox("Γλώσσα:", ["English", "German", "French"], key="trans_lang")
            if st.button("Εκτέλεση Μετάφρασης", key="btn_trans") and txt:
                with st.spinner("Μετάφραση..."):
                    res = llm.invoke(f"Act as Strict Legal Translator. Translate to {lang}. Output ONLY text. No notes.\nText: {txt}")
                    st.write(res.content)
        else:
            txt = st.text_area("Κείμενο με προσωπικά δεδομένα:", key="anon_input")
            if st.button("Εκτέλεση Ανωνυμοποίησης", key="btn_anon") and txt:
                with st.spinner("Επεξεργασία..."):
                    res = llm.invoke(f"Act as GDPR Officer. Replace Names/AFM with placeholders [ΟΝΟΜΑ]. Output ONLY text.\nText: {txt}")
                    st.code(res.content, language="text")

    with t4:
        st.header("Νομικός Βοηθός AI")
        main_chat, side_context = st.columns([3, 1])
        with side_context:
            uploaded_file = st.file_uploader("Προσθήκη Εγγράφου", type="pdf", key="unified_pdf_uploader")
            if uploaded_file:
                if "current_pdf_id" not in st.session_state or st.session_state.current_pdf_id != uploaded_file.name:
                    with st.spinner("Ανάγνωση..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path); docs = loader.load()
                        st.session_state.analysis_text = "\n".join([d.page_content for d in docs])
                        st.session_state.current_pdf_id = uploaded_file.name
                        os.unlink(tmp_path)
                st.success(f"✅ {uploaded_file.name}")
            else: st.session_state.analysis_text = ""

        with main_chat:
            for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
            if prompt := st.chat_input("Ερώτηση...", key="unified_chat"):
                st.session_state.messages.append({"role": "user", "content": prompt}); st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    try:
                        vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                        target_ids = [current_firm, "Public_Legal_Library"]
                        search_filter = {"firm_id": {"$in": target_ids}}
                        match = re.search(r'(?:άρθρο|αρθρο|Article)\s*:?\s*(\d+)', prompt, re.IGNORECASE)
                        if match: search_filter["article_id"] = {"$eq": match.group(1)}
                        retriever = vs.as_retriever(search_kwargs={'filter': search_filter, 'k': 8})
                        db_docs = retriever.invoke(prompt)
                        db_context = str(db_docs)
                        pdf_context = st.session_state.analysis_text[:20000] if st.session_state.analysis_text else ""
                        final_context = f"DATABASE RESULTS:\n{db_context}\n\nUPLOADED DOCUMENT:\n{pdf_context}"
                        
                        system_prompt = """Είσαι ένας έμπειρος Νομικός Σύμβουλος.
                        ΟΔΗΓΙΕΣ:
                        1. Αν ρωτάνε για ΣΥΓΚΕΚΡΙΜΕΝΟ ΑΡΘΡΟ, ΨΑΞΕ το κείμενο στα 'DATABASE RESULTS'.
                        2. Αν το βρεις, παράθεσέ το ακριβώς.
                        3. Αν ΔΕΝ το βρεις, ΠΡΟΣΕΧΕ: Μην μαντέψεις το κείμενο του νόμου. Πες 'Δεν βρέθηκε στη βάση' και μετά δώσε τη γενική νομική σου γνώση.
                        
                        FORMAT:
                        [Απάντηση]
                        |||SOURCE:[DOC] (αν από PDF)
                        |||SOURCE:[DB] (αν από Βάση)
                        |||SOURCE:[AI] (αν Γενική Γνώση)
                        
                        CONTEXT: {context}
                        QUESTION: {question}"""
                        
                        chain = ChatPromptTemplate.from_template(system_prompt) | llm | StrOutputParser()
                        full_response = chain.invoke({"context": final_context, "question": prompt})
                        if "|||SOURCE:" in full_response: ans, source_tag = full_response.split("|||SOURCE:")
                        else: ans, source_tag = full_response, "[UNKNOWN]"
                        
                        st.write(ans.strip()); st.session_state.messages.append({"role": "assistant", "content": ans.strip()})
                        with st.expander("Πηγές & Δεδομένα"):
                            if "[AI]" in source_tag: st.info("🧠 **AI Knowledge / Not Found in DB**")
                            elif "[DOC]" in source_tag: st.success("📄 **Uploaded Document**")
                            elif "[DB]" in source_tag: 
                                st.markdown("🗄️ **Βάση Δεδομένων (Public & Private):**")
                                for i, doc in enumerate(db_docs):
                                    fname = doc.metadata.get("file_name", "Unknown")
                                    art = f"[Art. {doc.metadata.get('article_id')}]" if doc.metadata.get('article_id') else ""
                                    st.caption(f"{i+1}. {fname} {art}")
                    except Exception as e: st.error(f"Error: {e}")

    # --- TAB 5: FIXED TEMPLATE PROMPT ---
    with t5:
        st.subheader("Αυτόματη Σύνταξη Εξωδίκου")
        st.caption("Συμπληρώστε τα στοιχεία και το σύστημα θα παραγάγει ένα αυστηρά δομημένο νομικό έγγραφο.")

        with st.form("eviction_draft_form"):
            col_owner, col_tenant = st.columns(2)
            with col_owner:
                st.markdown("### 🏠 Εκμισθωτής (Ιδιοκτήτης)")
                l_name = st.text_input("Ονοματεπώνυμο", placeholder="π.χ. Γεώργιος Παπαδόπουλος")
                l_father = st.text_input("Πατρώνυμο (Ιδιοκτήτη)", placeholder="π.χ. του Δημητρίου")
                l_afm = st.text_input("ΑΦΜ (Ιδιοκτήτη)", placeholder="π.χ. 000000000")
                l_address = st.text_input("Διεύθυνση Κατοικίας", placeholder="π.χ. Εγνατία 10, Θεσσαλονίκη")

            with col_tenant:
                st.markdown("### 👤 Μισθωτής (Ενοικιαστής)")
                t_name = st.text_input("Ονοματεπώνυμο", placeholder="π.χ. Νικόλαος Γεωργίου")
                t_father = st.text_input("Πατρώνυμο (Μισθωτή)", placeholder="π.χ. του Κωνσταντίνου")
                t_afm = st.text_input("ΑΦΜ (Μισθωτή)", placeholder="π.χ. 999999999")
                t_address = st.text_input("Διεύθυνση Μισθίου", placeholder="π.χ. Τσιμισκή 50, Θεσσαλονίκη")

            st.markdown("### 💰 Στοιχεία Οφειλής")
            c1, c2, c3 = st.columns(3)
            rent_amount = c1.number_input("Μηνιαίο Μίσθωμα (€)", min_value=0.0, step=10.0, format="%.2f")
            unpaid_months = c2.text_input("Μήνες Καθυστέρησης", placeholder="π.χ. Ιανουάριος & Φεβρουάριος 2024")
            doc_date = c3.date_input("Ημερομηνία Εγγράφου", datetime.date.today())

            submit_draft = st.form_submit_button("✍️ Σύνταξη Εγγράφου")

        if submit_draft:
            if not l_name or not t_name:
                st.warning("Παρακαλώ συμπληρώστε τουλάχιστον τα ονόματα.")
            else:
                with st.spinner("Δημιουργία Εγγράφου..."):
                    # THE ONE-SHOT PROMPT (Template based)
                    draft_prompt = f"""
                    Ενέργησε ως έμπειρος Έλληνας Δικηγόρος.
                    Στόχος: Σύνταξε μια επίσημη ΕΞΩΔΙΚΗ ΔΗΛΩΣΗ - ΠΡΟΣΚΛΗΣΗ - ΔΙΑΜΑΡΤΥΡΙΑ.
                    
                    ΔΕΔΟΜΕΝΑ:
                    - Εκμισθωτής (Καλών): {l_name} {l_father}, ΑΦΜ {l_afm}, κάτοικος {l_address}.
                    - Μισθωτής (Καθ' ου): {t_name} {t_father}, ΑΦΜ {t_afm}, κάτοικος {t_address} (Μίσθιο).
                    - Ποσό Μισθώματος: {rent_amount} Ευρώ.
                    - Οφειλόμενοι Μήνες: {unpaid_months}.
                    - Ημερομηνία: {doc_date}.

                    ΟΔΗΓΙΕΣ ΜΟΡΦΟΠΟΙΗΣΗΣ (ΑΚΟΛΟΥΘΗΣΕ ΑΥΣΤΗΡΑ):
                    1. Ξεκίνα το έγγραφο ΑΚΡΙΒΩΣ με τη φράση: "ΕΝΩΠΙΟΝ ΠΑΝΤΟΣ ΑΡΜΟΔΙΟΥ ΔΙΚΑΣΤΗΡΙΟΥ ΚΑΙ ΠΑΣΗΣ ΑΡΧΗΣ".
                    2. Τίτλος: "ΕΞΩΔΙΚΗ ΔΗΛΩΣΗ - ΠΡΟΣΚΛΗΣΗ - ΔΙΑΜΑΡΤΥΡΙΑ ΜΕ ΕΠΙΦΥΛΑΞΗ ΔΙΚΑΙΩΜΑΤΩΝ".
                    3. Μην γράψεις εισαγωγές τύπου "Ορίστε το έγγραφο". Δώσε μόνο το καθαρό νομικό κείμενο.
                    4. Χρησιμοποίησε επίσημη, νομική γλώσσα (καθαρεύουσα όπου είθισται, π.χ. "κοινοποιουμένη", "αιτούμαι").
                    5. Ανάφερε ρητά την προθεσμία των 15 ημερών (άρθρο 637 ΚΠολΔ / 597 ΑΚ).
                    6. Κλείσε με τόπο, ημερομηνία και "Ο Πληρεξούσιος Δικηγόρος".
                    """
                    
                    response = llm.invoke(draft_prompt)
                    st.markdown("### 📄 Παραγόμενο Έγγραφο")
                    st.text_area("Αντιγραφή Κειμένου (Copy-Paste σε Word):", value=response.content, height=600)

    with t6:
        st.subheader("Παρακολούθηση Προθεσμιών")
        with st.expander("Προσθήκη"):
            with st.form("w"):
                n = st.text_input("Όνομα", key="w_name"); e = st.text_input("Email", key="w_email"); d = st.number_input("Ποσό", key="w_debt"); sd = st.date_input("Ημερομηνία", key="w_date")
                if st.form_submit_button("Καταγραφή"): deadline = sd + datetime.timedelta(days=15); st.session_state.active_evictions.append({"id": len(st.session_state.active_evictions), "name": n, "email": e, "debt": d, "deadline": deadline, "status": "Pending"}); st.rerun()
        cases = st.session_state.active_evictions
        for c in cases:
            if c["status"] == "Pending":
                with st.container():
                    c1, c2, c3, c4 = st.columns([2,2,2,2])
                    c1.write(f"**{c['name']}**")
                    c2.write(f"Λήξη: {c['deadline']}")
                    if c3.button("Email", key=f"e_{c['id']}"):
                        show_email_draft(c['name'], c['email'], c['debt'], str(c['deadline']), current_firm)
                    if c4.button("Εξοφλήθη", key=f"p_{c['id']}"):
                        c["status"] = "Paid"; st.rerun()

if "logged_in" not in st.session_state: st.session_state['logged_in'] = False
if not st.session_state['logged_in']: login_page()
else: main_app()
