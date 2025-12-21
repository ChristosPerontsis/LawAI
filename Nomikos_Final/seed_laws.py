import json
import os
# Using Community embeddings for stability on your Mac
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# KEYS
os.environ["PINECONE_API_KEY"] = "pcsk_6jvSUB_EXMzV8C2PgeLxXfJifWQPAqa2W4EefL4QcD5DV2JyPSjtgAzoGnURQ6T7peqs5D"
INDEX_NAME = "nomikos-index"

print("🚀 Starting Smart Upload...")

# 1. Load Brain
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

# 2. Load JSON
try:
    with open("greek_laws.json", "r", encoding="utf-8") as f:
        laws = json.load(f)
except:
    print("❌ Error: greek_laws.json missing.")
    exit()

# 3. Prepare Documents with STRICT LIMITS
final_docs = []
# 2000 chars is safe (well below the 40KB limit)
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

print(f"   - Processing {len(laws)} articles...")

for law in laws:
    # Combine Title + Text
    content = f"Άρθρο {law['article']}: {law['title']}\n\n{law['text']}"
    
    # Check size and split if necessary
    # Pinecone limit is 40KB metadata. We keep chunks small to be safe.
    if len(content.encode('utf-8')) > 30000:
        print(f"   ⚠️ Article {law['article']} is too big ({len(content)} chars). Splitting...")
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "firm_id": "Public_Legal_Library",
                    "source_type": "public",
                    "article_id": f"{law['article']}_part_{i}",
                    "category": law['category']
                }
            )
            final_docs.append(doc)
    else:
        doc = Document(
            page_content=content,
            metadata={
                "firm_id": "Public_Legal_Library",
                "source_type": "public",
                "article_id": str(law['article']),
                "category": law['category']
            }
        )
        final_docs.append(doc)

print(f"   - Final count: {len(final_docs)} vectors ready.")

# 4. Upload
try:
    PineconeVectorStore.from_documents(final_docs, embeddings, index_name=INDEX_NAME)
    print("✅ UPLOAD COMPLETE! The AI Brain is updated.")
except Exception as e:
    print(f"❌ Upload Failed: {e}")
