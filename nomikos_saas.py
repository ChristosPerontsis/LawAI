import streamlit as st
import os
import tempfile
import datetime
import re
import json
import time
import hashlib
import pandas as pd
# --- SWAPPED IMPORTS: REMOVED GROQ, ADDED GOOGLE ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    # Hardcoded Admin for initial setup
    if username == "admin":
        return {"pass": hash_password("admin"), "firm_id": "ADMIN_Δημόσια_Βιβλιοθήκη", "role": "admin"}

    engine = get_db_connection()
    if engine:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM users WHERE username = :u"), {"u": username}).fetchone()
                if result:
                    return {"pass": result[1], "firm_id": result[2], "role": result[3]}
        except: pass
    
    # Local JSON Fallback
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
        except: return False

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
    
    t1, t2, t3, t4, t5, t6 = st.tabs(["Αρχειοθέτηση", "Διαχείριση Αρχείων", "Εργαλεία", "Νομικός Βοηθός", "Αυτόματη Σύνταξη", "Διαχείριση Εξώσεων"])
    
    with t1:
        st.header("Εισαγωγή Νέων Εγγράφων")
        
        # --- LOGIC SWITCH BASED ON USER ROLE ---
        if "ADMIN" in current_firm:
            st.info("🔓 **ADMIN MODE**: Τα αρχεία που ανεβάζετε εδώ θα είναι ορατά σε ΟΛΟΥΣ τους χρήστες (Public Library).")
            splitter_type = st.radio("Τύπος Εγγράφου:", ["Κώδικας/Νόμοι (Smart Article Splitter)", "Απλό Έγγραφο"], horizontal=True)
        else:
            splitter_type = "Απλό Έγγραφο"

        with st.container():
            files = st.file_uploader("Επιλέξτε αρχεία PDF", accept_multiple_files=True, key="uploader")
            if st.button("🔒 Αποθήκευση στη Βάση", key="btn_upload") and files:
                with st.spinner("Επεξεργασία & Καταχώρηση..."):
                    for f in files:
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(f.read())
                                path = tmp.name
                            
                            loader = PyPDFLoader(path)
                            docs = loader.load()
                            clean_name = f.name
                            
                            # --- FIXED SURGEON SPLITTER LOGIC ---
                            if splitter_type == "Κώδικας/Νόμοι (Smart Article Splitter)":
                                # Updated separators to handle "Άρθρο : 126" and "Άρθρο 126"
                                splitter = RecursiveCharacterTextSplitter(
                                    separators=[
                                        "\nΆρθρο :", "\nΑΡΘΡΟ :", "\nΆρθρο:", "\nΑΡΘΡΟ:", # With Colon
                                        "\nΆρθρο ", "\nΑΡΘΡΟ ", # Without Colon
                                        "Άρθρο :", "ΑΡΘΡΟ :", 
                                        "Άρθρο ", "ΑΡΘΡΟ ",
                                        "\n\n", "\n"
                                    ],
                                    chunk_size=2000, 
                                    chunk_overlap=50,
                                    keep_separator=True
                                )
                            else:
                                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                            
                            splits = splitter.split_documents(docs)
                            
                            # --- PUBLIC vs PRIVATE LOGIC ---
                            upload_type = "public" if "ADMIN" in current_firm else "private"
                            target_id = "Public_Legal_Library" if "ADMIN" in current_firm else current_firm
                            
                            for d in splits:
                                d.metadata["firm_id"] = target_id
                                d.metadata["source_type"] = upload_type
                                d.metadata["file_name"] = clean_name
                                
                                # FIXED REGEX to capture Article ID with optional colon
                                # Matches: "Άρθρο 126" OR "Άρθρο : 126" OR "Άρθρο:126"
                                match = re.search(r'(Άρθρο|ΑΡΘΡΟ)\s*:?\s*(\d+)', d.page_content)
                                if match:
                                    d.metadata["article_id"] = match.group(2)

                            PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
                            os.unlink(path)
                            st.session_state.current_focus_file = clean_name
                        except Exception as e: st.error(f"Error: {e}")
                st.success(f"Επιτυχία! Το αρχείο ανέβηκε στην {'Δημόσια' if 'ADMIN' in current_firm else 'Ιδιωτική'} Βιβλιοθήκη.")

    with t2:
        st.header("Διαχείριση Εγγράφων & Έλεγχος")
        col1, col2 = st.columns([3, 1])
        q = col1.text_input("Αναζήτηση (π.χ. 'Άρθρο 125')", key="file_search_input")
        if col2.button("Αναζήτηση", key="btn_file_search"):
            vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
            # SEARCH BOTH PRIVATE AND PUBLIC
            target_ids = [current_firm, "Public_Legal_Library"]
            
            res = vs.similarity_search(q, k=10, filter={"firm_id": {"$in": target_ids}})
            
            if not res: st.warning("Δεν βρέθηκαν αποτελέσματα.")
            
            for i, d in enumerate(res):
                fname = d.metadata.get("file_name", "Άγνωστο")
                fid = d.metadata.get("firm_id")
                # Show explicit Article ID if we found it
                art_tag = f" [Art. {d.metadata.get('article_id')}]" if d.metadata.get('article_id') else ""
                
                with st.expander(f"{i+1}. {fname}{art_tag} ({'PUBLIC' if 'Public' in fid else 'PRIVATE'})"):
                    st.text(d.page_content)
                    # Only Admin can delete Public files
                    if "Public" in fid and "ADMIN" not in current_firm:
                        st.caption("🔒 Read-only (Public Library)")
                    else:
                        if st.button("Διαγραφή", key=f"del_{i}"):
                            pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                            pc.Index(index_name).delete(filter={"firm_id": fid, "file_name": fname})
                            st.toast("Deleted")
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
        main_chat, side_context = st.columns([3, 1])
        with side_context:
            st.info("📂 **Ενεργό Έγγραφο**")
            uploaded_file = st.file_uploader("Προσθήκη Εγγράφου στη Συζήτηση", type="pdf", key="unified_pdf_uploader")
            if uploaded_file:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if "current_pdf_id" not in st.session_state or st.session_state.current_pdf_id != file_id:
                    with st.spinner("Ανάγνωση..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
                            loader = PyPDFLoader(tmp_path); docs = loader.load()
                            full_text = "\n".join([d.page_content for d in docs])
                            st.session_state.analysis_text = full_text; st.session_state.current_pdf_id = file_id; os.unlink(tmp_path)
                        except: pass
                st.success(f"✅ {uploaded_file.name}")
            else: st.session_state.analysis_text = ""; st.markdown("*Κανένα ενεργό έγγραφο.*")

        with main_chat:
            for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
            if prompt := st.chat_input("Πληκτρολογήστε την ερώτησή σας...", key="unified_chat"):
                st.session_state.messages.append({"role": "user", "content": prompt}); st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    try:
                        vs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                        # SEARCH: Current Firm + Public Library
                        target_ids = [current_firm, "Public_Legal_Library"]
                        search_filter = {"firm_id": {"$in": target_ids}}
                        
                        retriever = vs.as_retriever(search_kwargs={'filter': search_filter, 'k': 6}) # Increase k to find right article
                        db_docs = retriever.invoke(prompt)
                        db_context = str(db_docs)
                        pdf_context = st.session_state.analysis_text[:20000] if st.session_state.analysis_text else ""
                        final_context = f"DATABASE RESULTS:\n{db_context}\n\nUPLOADED DOCUMENT:\n{pdf_context}"
                        
                        system_prompt = """Είσαι ένας έμπειρος Νομικός Σύμβουλος.
                        
                        ΟΔΗΓΙΕΣ ΓΙΑ ΑΡΘΡΑ ΝΟΜΩΝ:
                        1. Αν ρωτάνε για άρθρο (π.χ. 125), ΕΛΕΓΞΕ ΤΑ 'DATABASE RESULTS' για κείμενο που ξεκινά με 'Άρθρο : 125' ή 'Άρθρο 125'.
                        2. Αν το βρεις, παράθεσέ το ακριβώς.
                        3. Αν ΔΕΝ το βρεις, πες 'Δεν βρέθηκε στη βάση' και μετά δώσε τη γενική γνώση σου.
                        
                        FORMAT:
                        [Απάντηση]
                        
                        [Κενή Γραμμή]
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

    with t5:
        st.subheader("Αυτόματη Σύνταξη Εξωδίκου")
        with st.form("draft"):
            c1, c2, c3 = st.columns(3)
            l_name = c1.text_input("Εκμισθωτής", key="l_name")
            l_father = c2.text_input("Πατρώνυμο", key="l_father")
            l_afm = c3.text_input("ΑΦΜ", key="l_afm")
            l_addr = st.text_input("Διεύθυνση", key="l_addr")
            
            t1_col, t2_col, t3_col = st.columns(3)
            t_name = t1_col.text_input("Μισθωτής", key="t_name")
            t_father = t2_col.text_input("Πατρώνυμο", key="t_father")
            t_afm = t3_col.text_input("ΑΦΜ", key="t_afm")
            
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
        st.subheader("Παρακολούθηση Προθεσμιών")
        with st.expander("Προσθήκη"):
            with st.form("w"):
                n = st.text_input("Όνομα", key="w_name")
                e = st.text_input("Email", key="w_email")
                d = st.number_input("Ποσό", key="w_debt")
                sd = st.date_input("Ημερομηνία", key="w_date")
                
                if st.form_submit_button("Καταγραφή"):
                    deadline = sd + datetime.timedelta(days=15)
                    st.session_state.active_evictions.append({
                        "id": len(st.session_state.active_evictions),
                        "name": n, 
                        "email": e, 
                        "debt": d, 
                        "deadline": deadline, 
                        "status": "Pending"
                    })
                    st.rerun()
        
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
                        c["status"] = "Paid"
                        st.rerun()

if "logged_in" not in st.session_state: st.session_state['logged_in'] = False
if not st.session_state['logged_in']: login_page()
else: main_app()
