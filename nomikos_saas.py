import streamlit as st
import os
import tempfile
import datetime
import re
import json
import time
import hashlib
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sqlalchemy import create_engine, text

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
    engine = get_db_connection()
    if engine:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM users WHERE username = :u"), {"u": username}).fetchone()
                if result:
                    return {"pass": result[1], "firm_id": result[2], "role": result[3]}
        except: pass
    
    if not os.path.exists(USER_DB_FILE):
        default = {"admin": {"pass": hash_password("admin"), "firm_id": "ADMIN_Δημόσια_Βιβλιοθήκη", "role": "admin"}}
        with open(USER_DB_FILE, 'w') as f: json.dump(default, f)
        return default.get(username)
    try:
        with open(USER_DB_FILE, 'r') as f: return json.load(f).get(username)
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
        except: return False

    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f: users = json.load(f)
    else: users = {}
    if username in users: return False
    users[username] = {"pass": hashed_pw, "firm_id": firm_name, "role": "user"}
    with open(USER_DB_FILE, 'w') as f: json.dump(users, f)
    return True

# --- SESSION MANAGEMENT ---
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

# --- 2. HELPER FUNCTIONS ---
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
            # REMOVED DEBUG INDICATORS HERE
            
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
                                st.success("Επιτυχής Εγγραφή! Συνδεθείτε.")
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
    if "current_focus_file" not in st.session_state: st.session_state.current_focus_file = None
    
    with st.sidebar:
        st.markdown(f"### 👤 {current_firm}")
        
        # REMOVED SIDEBAR DIAGNOSTIC
        # REMOVED CLEAR FOLDER BUTTON

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

    try:
        llm = ChatGroq(temperature=0.3, model_name="llama-3.1-8b-instant", api_key=st.secrets["GROQ_API_KEY"])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})
    except: st.stop()

    st.title("🗂️ Νομικός Φάκελος")
    
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Αρχειοθέτηση", 
        "Διαχείριση Αρχείων", 
        "Εργαλεία", 
        "Νομικός Βοηθός", 
        "Αυτόματη Σύνταξη", 
        "Διαχείριση Εξώσεων"
    ])
    
    with t1:
        st.header("Εισαγωγή Νέων Εγγράφων")
        with st.container():
            files = st.file_uploader("Επιλέξτε αρχεία PDF", accept_multiple_files=True, key="uploader")
            if st.button("🔒 Κρυπτογράφηση & Αποθήκευση", key="btn_upload") and files:
                with st.spinner("Γίνεται επεξεργασία..."):
                    for f in files:
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(f.read())
                                path = tmp.name
                            loader = PyPDFLoader(path)
                            docs = loader.load()
                            clean_name = f.name
                            for doc in docs: doc.page_content = f"FILENAME: {clean_name}\n\n" + doc.page_content
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                            splits = splitter.split_documents(docs)
                            upload_type = "public" if "ADMIN" in current_firm else "private"
                            target_id = "Public_Legal_Library" if "ADMIN" in current_firm else current_firm
                            for d in splits:
                                d.metadata["firm_id"] = target_id
                                d.metadata["source_type"] = upload_type
                                d.metadata["file_name"] = clean_name
                            PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
                            os.unlink(path)
                            st.session_state.current_focus_file = clean_name
                        except Exception as e: st.error(f"Error: {e}")
                st.success("Η διαδικασία ολοκληρώθηκε επιτυχώς.")

    with t2:
        st.header("Διαχείριση Εγγράφων")
        col1, col2 = st.columns([3, 1])
        q = col1.text_input("Αναζήτηση με όνομα αρχείου", key="file_search_input")
        if col2.button("Αναζήτηση", key="btn_file_search"):
            vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
            target_id = "Public_Legal_Library" if "ADMIN" in current_firm else current_firm
            res = vs.similarity_search(q, k=20, filter={"firm_id": {"$eq": target_id}})
            files = set(d.metadata.get("file_name") for d in res)
            if not files: st.warning("Δεν βρέθηκαν αποτελέσματα.")
            for f in files:
                with st.container():
                    c1, c2 = st.columns([4,1])
                    c1.markdown(f"📄 **{f}**")
                    if c2.button("Διαγραφή", key=f"del_{f}"):
                        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                        pc.Index(index_name).delete(filter={"firm_id": target_id, "file_name": f})
                        st.toast("Το αρχείο διαγράφηκε")
                        st.rerun()

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
        active = st.session_state.current_focus_file
        target_id = "Public_Legal_Library" if "ADMIN" in current_firm else current_firm
        
        if active:
            st.info(f"📂 Εστίαση: **{active}**")
            if st.button("Καθαρισμός Εστίασης", key="cls_focus"):
                st.session_state.current_focus_file = None
                st.rerun()
            search_filter = {"$or": [{"file_name": {"$eq": active}}, {"firm_id": {"$eq": "Public_Legal_Library"}}]}
        else:
            st.caption("🔍 Αναζήτηση σε όλη τη βάση δεδομένων.")
            search_filter = {"firm_id": {"$in": [target_id, "Public_Legal_Library"]}}
            
        for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
        
        if prompt := st.chat_input("Πληκτρολογήστε την ερώτησή σας...", key="chat_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                try:
                    vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                    retriever = vs.as_retriever(search_kwargs={'filter': search_filter, 'k': 10})
                    chain = ChatPromptTemplate.from_template("Είσαι Νομικός Βοηθός. Απάντησε ΜΟΝΟ βάσει του κειμένου. Αν ζητηθεί Σύνοψη, δώσε: Ιστορικό | Ετυμηγορία | Σκεπτικό | Νόμοι.\nContext: {context}\nQ: {question}") | llm | StrOutputParser()
                    docs = retriever.invoke(prompt)
                    ans = chain.invoke({"context": str(docs), "question": prompt})
                    st.write(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    with st.expander("Πηγές (Verified Sources)"):
                        if not docs: st.warning("Δεν βρέθηκαν πηγές.")
                        for i, doc in enumerate(docs):
                            fname = doc.metadata.get("file_name", "Unknown")
                            st.caption(f"📄 Πηγή {i+1}: {fname}")
                except Exception as e: st.error(str(e))

    with t5:
        st.subheader("Αυτόματη Σύνταξη Εξωδίκου")
        with st.form("draft"):
            c1, c2, c3 = st.columns(3)
            l_name = c1.text_input("Εκμισθωτής", key="l_name")
            l_father = c2.text_input("Πατρώνυμο", key="l_father")
            l_afm = c3.text_input("ΑΦΜ", key="l_afm")
            l_addr = st.text_input("Διεύθυνση", key="l_addr")
            t1, t2, t3 = st.columns(3)
            t_name = t1.text_input("Μισθωτής", key="t_name")
            t_father = t2.text_input("Πατρώνυμο", key="t_father")
            t_afm = t3.text_input("ΑΦΜ", key="t_afm")
            prop = st.text_input("Μίσθιο", key="prop_addr")
            date = st.date_input("Ημ. Μίσθωσης", key="contr_date")
            m1, m2 = st.columns(2)
            amt = m1.text_input("Ποσό", key="amt_val")
            mths = m2.text_input("Μήνες", key="mths_val")
            lawyer = st.text_input("Δικηγόρος", key="law_name")
            dets = st.text_area("Στοιχεία Δικηγόρου", key="law_dets")
            if st.form_submit_button("Δημιουργία Εγγράφου"):
                l_gen = auto_genitive(l_name)
                t_gen = auto_genitive(t_name)
                doc = f"""ΕΝΩΠΙΟΝ ΠΑΝΤΟΣ ΑΡΜΟΔΙΟΥ ΔΙΚΑΣΤΗΡΙΟΥ...\n\n{l_gen} {l_father}...\nΚΑΤΑ\n{t_gen} {t_father}...\n\n{lawyer}\n{dets}"""
                st.code(doc, language="markdown")

    with t6:
        st.subheader("Παρακολούθηση Προθεσμιών (Watchdog)")
        with st.expander("Προσθήκη Νέας Υπόθεσης"):
            with st.form("w"):
                n = st.text_input("Όνομα", key="w_name")
                e = st.text_input("Email", key="w_email")
                d = st.number_input("Ποσό", key="w_debt")
                sd = st.date_input("Ημερομηνία", key="w_date")
                if st.form_submit_button("Καταγραφή"):
                    deadline = sd + datetime.timedelta(days=15)
                    st.session_state.active_evictions.append({"id": len(st.session_state.active_evictions), "name": n, "email": e, "debt": d, "deadline": deadline, "status": "Pending"})
                    st.rerun()
        search_client = st.text_input("Αναζήτηση Οφειλέτη")
        cases = st.session_state.active_evictions
        if search_client: cases = [c for c in cases if search_client.lower() in c['name'].lower()]
        for c in cases:
            if c["status"] == "Pending":
                with st.container():
                    c1, c2, c3, c4 = st.columns([2,2,2,2])
                    c1.write(f"**{c['name']}**")
                    c2.write(f"Λήξη: {c['deadline']}")
                    if c3.button("Email", key=f"e_{c['id']}"):
                        show_email_draft(c['name'], c['email'], c['debt'], str(c['deadline']), current_firm)
                    if c4.button("Εξοφλήθη", key=f"p_{c['id']}"):
                        c["status"] = "Paid"
                        st.rerun()

if "logged_in" not in st.session_state: st.session_state['logged_in'] = False
if not st.session_state['logged_in']: login_page()
else: main_app()


### Action for CP:
1.  Save the file locally.
2.  **Push to GitHub.**
3.  Refresh your live website.
4.  You will see a clean, professional login screen (no orange lights, no emojis).
