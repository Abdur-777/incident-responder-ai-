import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# === LOAD ENV ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="IncidentResponder AI", page_icon="🚨", layout="wide")
st.title("🚨 IncidentResponder AI – Basic Plan")
st.caption("For councils & government – AI-powered incident & complaint responses")

# === SESSION STATE ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
if "role" not in st.session_state:
    st.session_state.role = "user"

# === HELPER FUNCTIONS ===
def build_pdf_index(pdf_dir, index_path):
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def load_faiss_index(index_path):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)
    return None

def ai_qa(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=False)
    resp = chain({"query": question})
    return resp.get("result", "No answer found.")

# === SIDEBAR NAVIGATION ===
menu = st.sidebar.radio("Navigation", ["Chat", "Upload Docs (Admin)", "About"])
if menu == "Chat":
    st.subheader("💬 Chat with IncidentResponder AI")
    if st.session_state.pdf_index is None:
        st.warning("No council incident/complaint docs uploaded yet.")
    else:
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {msg}")
        user_q = st.text_input("Ask a question...")
        if st.button("Send"):
            st.session_state.chat_history.append(("You", user_q))
            reply = ai_qa(user_q, st.session_state.pdf_index)
            st.session_state.chat_history.append(("AI", reply))
            st.experimental_rerun()

elif menu == "Upload Docs (Admin)":
    st.subheader("📄 Admin – Upload PDFs")
    if st.session_state.role != "admin":
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASS:
                st.session_state.role = "admin"
                st.success("Admin access granted.")
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")
    else:
        uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_pdfs:
            pdf_dir = "pdfs"
            os.makedirs(pdf_dir, exist_ok=True)
            for pdf in uploaded_pdfs:
                with open(os.path.join(pdf_dir, pdf.name), "wb") as f:
                    f.write(pdf.getbuffer())
            st.session_state.pdf_index = build_pdf_index(pdf_dir, "index/faiss_index")
            st.success("PDFs processed & AI ready to respond.")

elif menu == "About":
    st.subheader("ℹ️ About IncidentResponder AI")
    st.write("""
    This is the **Basic Plan ($500/month)** for councils & government agencies.
    - Upload incident & complaint handling policies
    - AI auto-answers resident queries
    - No per-user limits
    - Simple admin panel for PDF upload
    """)

st.markdown("<br><center>Made with ❤️ for Councils</center>", unsafe_allow_html=True)
