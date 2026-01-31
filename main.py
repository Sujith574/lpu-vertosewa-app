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
def chunk_text(text, chunk_size=350):
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
# LOAD ADMIN DOCUMENTS
# ------------------------------------------------------
def load_admin_documents():
    db = get_db()
    documents = []

    snaps = (
        db.collection("lpu_content")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(150)
        .stream()
    )

    for snap in snaps:
        d = snap.to_dict()
        for chunk in chunk_text(d.get("textContent", "")):
            documents.append({
                "source": "Admin Dashboard",
                "title": d.get("title", ""),
                "text": chunk,
                "embedding": embed_text(chunk)
            })

    return documents

# ------------------------------------------------------
# LOAD STATIC DOCUMENTS
# ------------------------------------------------------
def load_static_documents():
    documents = []
    for chunk in chunk_text(STATIC_LPU):
        documents.append({
            "source": "LPU Knowledge Base",
            "title": "lpu_knowledge.txt",
            "text": chunk,
            "embedding": embed_text(chunk)
        })
    return documents

# ------------------------------------------------------
# SEMANTIC SEARCH
# ------------------------------------------------------
def semantic_search(query, documents, top_k=4, threshold=0.35):
    query_vec = embed_text(query)
    scored = []

    for doc in documents:
        score = cosine_similarity(
            [query_vec], [doc["embedding"]]
        )[0][0]
        if score >= threshold:
            scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:top_k]]

# ------------------------------------------------------
# MEMORY (SESSION BASED)
# ------------------------------------------------------
conversation_memory = {}

def update_memory(session_id, role, content):
    conversation_memory.setdefault(session_id, []).append(
        {"role": role, "content": content}
    )
    conversation_memory[session_id] = conversation_memory[session_id][-6:]

# ------------------------------------------------------
# WELCOME MESSAGE
# ------------------------------------------------------
def welcome_message():
    return (
        "Hello 👋 Welcome to **LPU VertoSewa**.\n\n"
        "I’m an AI assistant for **Lovely Professional University (LPU)**.\n\n"
        "You can ask me about:\n"
        "• Academics, exams, attendance\n"
        "• Hostels, fees, discipline\n"
        "• UMS / RMS / DSW notices\n"
        "• General questions as well\n\n"
        "How can I help you today?"
    )

# ------------------------------------------------------
# TIME & DATE
# ------------------------------------------------------
def handle_time_date(text):
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if text in ["time", "time now"]:
        return f"⏰ {now.strftime('%I:%M %p')} (IST)"

    if text in ["date", "today date", "date today"]:
        return f"📅 {now.strftime('%d %B %Y')}"

    return None

# ------------------------------------------------------
# GEMINI – GENERAL
# ------------------------------------------------------
def gemini_general(question):
    prompt = f"""
Answer clearly using general public knowledge.
Reply only in English.

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
# GEMINI – CONTEXTUAL (LPU)
# ------------------------------------------------------
def gemini_contextual(question, context, memory):
    prompt = f"""
You are an official AI assistant for Lovely Professional University (LPU).

Conversation history:
{memory}

Use ONLY the following verified context.
If answer is missing, say so clearly.

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

    # 1️⃣ First message → Welcome
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
        update_memory(session_id, "assistant", welcome_message())
        return welcome_message()

    # 2️⃣ Greetings (NO Gemini)
    if text in ["hi", "hii", "hello", "hey", "hai"]:
        return "Hello 👋 How can I help you?"

    # 3️⃣ Time / Date
    td = handle_time_date(text)
    if td:
        return td

    # 4️⃣ Load documents
    admin_docs = load_admin_documents()
    static_docs = load_static_documents()
    all_docs = admin_docs + static_docs

    # 5️⃣ Semantic search
    relevant = semantic_search(message, all_docs)

    if relevant:
        context = ""
        sources = set()

        for doc in relevant:
            context += f"\n[{doc['source']} – {doc['title']}]\n{doc['text']}\n"
            sources.add(f"{doc['source']} ({doc['title']})")

        memory = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in conversation_memory.get(session_id, [])
        )

        answer = gemini_contextual(message, context, memory)
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

        update_memory(session_id, "user", message)
        update_memory(session_id, "assistant", answer)
        return answer

    # 6️⃣ FINAL FALLBACK → GENERAL GEMINI
    answer = gemini_general(message)
    update_memory(session_id, "user", message)
    update_memory(session_id, "assistant", answer)
    return answer

# ------------------------------------------------------
# API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return {"reply": "Please enter a valid question."}

    return {"reply": process_message(session_id, message)}