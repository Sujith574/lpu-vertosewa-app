from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from google.cloud import firestore
from google import genai

# ------------------------------------------------------
# APP INIT
# ------------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------
# GEMINI CONFIG
# ------------------------------------------------------
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "models/gemini-2.5-flash"
EMBED_MODEL = "models/embedding-001"

# ------------------------------------------------------
# FIRESTORE
# ------------------------------------------------------
def get_db():
    return firestore.Client()

# ------------------------------------------------------
# LOAD STATIC LPU KNOWLEDGE
# ------------------------------------------------------
def load_lpu_knowledge():
    try:
        with open("lpu_knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

STATIC_LPU = load_lpu_knowledge()

# ------------------------------------------------------
# CHUNKING
# ------------------------------------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# ------------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------------
def embed_text(text: str):
    client = get_gemini_client()
    res = client.models.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return np.array(res["embedding"])

# ------------------------------------------------------
# VECTOR SEARCH
# ------------------------------------------------------
def semantic_search(query, documents, top_k=5):
    query_vec = embed_text(query)
    scored = []

    for doc in documents:
        score = cosine_similarity(
            [query_vec], [doc["embedding"]]
        )[0][0]
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:top_k]]

# ------------------------------------------------------
# LOAD ADMIN DASHBOARD CONTENT
# ------------------------------------------------------
def load_admin_documents():
    db = get_db()
    docs = []

    snapshots = (
        db.collection("lpu_content")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(100)
        .stream()
    )

    for snap in snapshots:
        d = snap.to_dict()
        chunks = chunk_text(d.get("textContent", ""))

        for chunk in chunks:
            docs.append({
                "source": "Admin Dashboard",
                "title": d.get("title", ""),
                "text": chunk,
                "embedding": embed_text(chunk)
            })

    return docs

# ------------------------------------------------------
# LOAD STATIC KNOWLEDGE DOCUMENTS
# ------------------------------------------------------
def load_static_documents():
    docs = []
    chunks = chunk_text(STATIC_LPU)

    for chunk in chunks:
        docs.append({
            "source": "LPU Knowledge Base",
            "title": "lpu_knowledge.txt",
            "text": chunk,
            "embedding": embed_text(chunk)
        })

    return docs

# ------------------------------------------------------
# SESSION MEMORY
# ------------------------------------------------------
conversation_memory = {}

def update_memory(session_id, role, content):
    conversation_memory.setdefault(session_id, []).append(
        {"role": role, "content": content}
    )
    conversation_memory[session_id] = conversation_memory[session_id][-6:]

# ------------------------------------------------------
# WELCOME MESSAGE (ONCE PER SESSION)
# ------------------------------------------------------
def get_welcome_message():
    return (
        "Hello 👋 Welcome to (LPU VertoSewa).\n\n"
        "I’m an AI assistant for **Lovely Professional University (LPU)**.\n\n"
        "You can ask me about:\n"
        "• Academics, exams, attendance\n"
        "• Hostels, fees, discipline\n"
        "• UMS / RMS / DSW notices\n"
        "• General questions as well\n"
    )

# ------------------------------------------------------
# TIME & DATE
# ------------------------------------------------------
def handle_time_date(text):
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if text in ["time", "time now"]:
        return f"⏰ {now.strftime('%I:%M %p')} (IST)"

    if text in ["date", "date today"]:
        return f"📅 {now.strftime('%d %B %Y')}"

    return None

# ------------------------------------------------------
# GEMINI ANSWER
# ------------------------------------------------------
def gemini_answer(question, context, memory):
    prompt = f"""
You are an official AI assistant for Lovely Professional University (LPU).

Conversation history:
{memory}

Use the following VERIFIED CONTEXT to answer.
Cite sources at the end.

CONTEXT:
{context}

QUESTION:
{question}
"""
    client = get_gemini_client()
    res = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return res.text.strip()

# ------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------
def process_message(session_id, message):
    text = message.lower().strip()

    # 🟢 WELCOME MESSAGE (FIRST MESSAGE ONLY)
    if session_id not in conversation_memory:
        welcome = get_welcome_message()
        update_memory(session_id, "assistant", welcome)
        return welcome

    # Time / Date
    td = handle_time_date(text)
    if td:
        return td

    # Load documents
    admin_docs = load_admin_documents()
    static_docs = load_static_documents()
    all_docs = admin_docs + static_docs

    # Semantic search
    relevant = semantic_search(message, all_docs)

    # Build context + citations
    context = ""
    sources = set()

    for doc in relevant:
        context += f"\n[{doc['source']} – {doc['title']}]\n{doc['text']}\n"
        sources.add(f"{doc['source']} ({doc['title']})")

    # Conversation memory
    memory = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation_memory.get(session_id, [])
    )

    answer = gemini_answer(message, context, memory)

    # Append sources
    answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    update_memory(session_id, "user", message)
    update_memory(session_id, "assistant", answer)

    return answer

# ------------------------------------------------------
# API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not message.strip():
        return {"reply": "Please enter a valid question."}

    return {"reply": process_message(session_id, message)}