import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from groq import Groq
from googlesearch import search
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI(title="NunAI Backend API")

# Initialize Groq Client & Vector Model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load Database & Setup FAISS Index
DB_FILE = "database.json"

def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

database_data = load_database()
dimension = 384
index = faiss.IndexFlatL2(dimension)

def update_faiss_index():
    global index
    index = faiss.IndexFlatL2(dimension)
    if database_data:
        texts = [f"{item.get('topik', '')}: {item.get('konten', '')}" for item in database_data]
        embeddings = embedder.encode(texts)
        index.add(np.array(embeddings).astype('float32'))

update_faiss_index()

# System Prompt & Persona NunAI
SYSTEM_PROMPT = """
Kamu adalah NunAI, sebuah Asisten AI Muslim Open Source dari Indonesia yang santun, bijak, dan penuh hangat.
Identitas Utama:
- Kamu dikembangkan oleh AI Studio (HANYA sebutkan nama AI Studio jika pengguna bertanya secara eksplisit siapa pembuatmu).
- Kamu dilatih dari ribuan hingga jutaan baris data tambahan Islami, sejarah Nabi, ajaran ulama, serta data model dasar Meta (Llama 3.3).
- Gaya bicara: Islami, sejuk seperti membawakan dakwah yang merangkul, mengedepankan ucapan salam, syukur, serta doa kebaikan.

Aturan Format & Cara Menjawab:
1. Berikan jawaban terstruktur menggunakan poin '●' untuk poin penting dan penomoran ' 1. ', ' 2. ', ' 3. ' untuk langkah-langkah.
2. Sediakan emoji yang relevan agar percakapan terasa hidup dan interaktif.
3. Teliti dan utamakan kebenaran data RAG/Database lokal yang diberikan. Jika data lokal tidak mencukupi, padukan dengan pengetahuan bawaan model atau pencarian web.
4. Kamu juga mahir dalam koding (semua jenis bahasa pemrograman) dan bisa menuliskan kode secara bersih dan terstruktur.
5. Toleransi Beragama: Jika pengguna bertanya tentang agama lain, jawablah dengan penuh rasa hormat dan toleransi. Sisipkan hikmah/ajaran Islam yang relevan secara halus dan inklusif agar menyentuh hati tanpa kesan menggurui.
6. Kemampuan Bahasa: Mampu mentranslate konteks data ke bahasa apa pun sesuai permintaan pengguna.
"""

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    web_search: Optional[bool] = False

def search_rag_database(query: str, top_k: int = 2) -> str:
    if not database_data or index.ntotal == 0:
        return ""
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(database_data):
            item = database_data[idx]
            results.append(f"- {item.get('topik', '')}: {item.get('konten', '')}")
    return "\n".join(results)

def google_web_search(query: str, num_results: int = 3) -> str:
    try:
        search_results = []
        for url in search(query, num_results=num_results, lang="id"):
            search_results.append(url)
        return "\n".join(search_results)
    except Exception:
        return ""

def auto_learn_and_save(user_query: str, ai_response: str):
    """
    Sistem belajar mandiri: Menyimpan intisari interaksi secara otomatis 
    ke memori database lokal FAISS untuk dipelajari kembali di masa depan.
    """
    global database_data
    new_entry = {
        "topik": f"Memori Pembelajaran: {user_query[:30]}...",
        "konten": f"Pertanyaan: {user_query} | Jawaban Ringkas: {ai_response[:200]}..."
    }
    database_data.append(new_entry)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(database_data, f, ensure_ascii=False, indent=2)
    update_faiss_index()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        context_data = search_rag_database(request.message)
        web_data = ""
        
        if request.web_search:
            web_data = google_web_search(request.message)
            
        augmented_prompt = f"[Konteks Database Internal NunAI]:\n{context_data}\n\n"
        if web_data:
            augmented_prompt += f"[Hasil Pencarian Google Realtime]:\n{web_data}\n\n"
        augmented_prompt += f"[Pesan Pengguna]: {request.message}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Memuat riwayat percakapan sebelumnya agar AI ingat topik
        for msg in request.history:
            messages.append({"role": msg.get("role"), "content": msg.get("content")})
            
        messages.append({"role": "user", "content": augmented_prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.6,
            max_tokens=2048
        )

        ai_reply = response.choices[0].message.content
        
        # Proses AI belajar mandiri menyimpan data pengetahuan baru
        auto_learn_and_save(request.message, ai_reply)

        return {
            "status": "success",
            "reply": ai_reply,
            "rag_used": bool(context_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Server Backend NunAI Berhasil Berjalan!"}
