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
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "models/gemini-2.5-flash"
EMBED_MODEL = "models/embedding-001"

# ------------------------------------------------------
# FIRESTORE
# ------------------------------------------------------
db = firestore.Client()

# ------------------------------------------------------
# MEMORY
# ------------------------------------------------------
conversation_memory = {}

# ------------------------------------------------------
# WELCOME
# ------------------------------------------------------
def welcome_message():
    return (
        "Hello 👋 Welcome to (LPU VertoSewa).\n\n"
        "Ask me anything related to Lovely Professional University (LPU).\n"
        "Academics • Hostels • Exams • UMS • DSW • General queries"
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
# EMBEDDING (SAFE)
# ------------------------------------------------------
def embed(text):
    try:
        res = client.models.embed_content(
            model=EMBED_MODEL,
            content=text
        )
        return np.array(res["embedding"])
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return None

# ------------------------------------------------------
# LOAD STATIC KNOWLEDGE (ONCE)
# ------------------------------------------------------
STATIC_DOCS = []

try:
    with open("lpu_knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()
        STATIC_DOCS.append({
            "source": "LPU Knowledge Base",
            "title": "lpu_knowledge.txt",
            "text": text,
            "embedding": embed(text)
        })
except:
    pass

# ------------------------------------------------------
# LOAD ADMIN CONTENT (ONCE)
# ------------------------------------------------------
ADMIN_DOCS = []

try:
    snaps = db.collection("lpu_content").stream()
    for s in snaps:
        d = s.to_dict()
        txt = d.get("textContent", "")
        ADMIN_DOCS.append({
            "source": "Admin Dashboard",
            "title": d.get("title", ""),
            "text": txt,
            "embedding": embed(txt)
        })
except Exception as e:
    logging.error(f"Firestore load error: {e}")

ALL_DOCS = [d for d in (ADMIN_DOCS + STATIC_DOCS) if d["embedding"] is not None]

# ------------------------------------------------------
# SEARCH
# ------------------------------------------------------
def semantic_search(query, top_k=4):
    q_vec = embed(query)
    if q_vec is None:
        return []

    scored = []
    for d in ALL_DOCS:
        score = cosine_similarity([q_vec], [d["embedding"]])[0][0]
        scored.append((score, d))

    scored.sort(reverse=True)
    return [d for _, d in scored[:top_k]]

# ------------------------------------------------------
# GEMINI ANSWER
# ------------------------------------------------------
def gemini_answer(question, context):
    prompt = f"""
Answer accurately using the context below.
If unsure, say you are unsure.

CONTEXT:
{context}

QUESTION:
{question}
"""
    try:
        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return res.text.strip()
    except Exception as e:
        logging.error(e)
        return "Sorry, I am facing a temporary issue. Please try again."

# ------------------------------------------------------
# CORE
# ------------------------------------------------------
def process(session_id, message):
    text = message.lower().strip()

    # Welcome (only once)
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
        return welcome_message()

    # Time/date
    td = handle_time_date(text)
    if td:
        return td

    docs = semantic_search(message)

    context = ""
    sources = set()

    for d in docs:
        context += f"\n[{d['source']} – {d['title']}]\n{d['text']}\n"
        sources.add(f"{d['source']} ({d['title']})")

    answer = gemini_answer(message, context)

    if sources:
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

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
        return {"reply": "Please enter a message."}

    return {"reply": process(session_id, message)}