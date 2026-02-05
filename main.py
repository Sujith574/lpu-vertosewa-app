from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz

from google.cloud import firestore
from google import genai

# ------------------------------------------------------
# APP INIT
# ------------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# ------------------------------------------------------
# GEMINI CONFIG
# ------------------------------------------------------
GEMINI_MODEL = "models/gemini-2.5-flash"

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

# ------------------------------------------------------
# FIRESTORE
# ------------------------------------------------------
def get_db():
    return firestore.Client()

# ------------------------------------------------------
# LOAD LPU KNOWLEDGE
# ------------------------------------------------------
def load_lpu_knowledge():
    try:
        with open("lpu_knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        logging.warning("lpu_knowledge.txt not found")
        return ""

STATIC_LPU = load_lpu_knowledge()

# ------------------------------------------------------
# SEARCH LPU KNOWLEDGE FILE
# ------------------------------------------------------
def search_lpu_knowledge(question: str, knowledge: str) -> str:
    q_words = question.lower().split()
    chunks = knowledge.split("\n\n")
    matches = []

    for chunk in chunks:
        chunk_l = chunk.lower()
        score = sum(1 for w in q_words if w in chunk_l)
        if score >= 3:
            matches.append((score, chunk))

    matches.sort(reverse=True, key=lambda x: x[0])

    if matches:
        return "\n\n".join(m[1] for m in matches[:2])

    return ""

# ------------------------------------------------------
# SEARCH ADMIN CONTENT (FIRESTORE)
# ------------------------------------------------------
def search_admin_content(question: str):
    db = get_db()
    q = question.lower()
    results = []

    docs = (
        db.collection("lpu_content")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(50)
        .stream()
    )

    for doc in docs:
        d = doc.to_dict()
        text = (d.get("textContent") or "").lower()
        keywords = [k.lower() for k in (d.get("keywords") or [])]
        category = (d.get("category") or "").lower()

        score = 0
        for k in keywords:
            if k in q:
                score += 2
        if category and category in q:
            score += 1

        if score > 0:
            results.append((score, d.get("textContent", "")))

    results.sort(reverse=True, key=lambda x: x[0])

    return "\n\n".join(r[1] for r in results[:2])

# ------------------------------------------------------
# GREETING
# ------------------------------------------------------
def handle_greeting(text: str):
    if any(text.startswith(g) for g in ["hi", "hello", "hey", "hai", "namaste"]):
        return (
            "Hello 👋 I’m **LPU VertoSewa**, the official AI assistant for "
            "**Lovely Professional University**.\n\n"
            "Ask me anything related to academics, exams, UMS/RMS, hostels, "
            "fees, or university policies."
        )
    return None

# ------------------------------------------------------
# TIME & DATE
# ------------------------------------------------------
def handle_time_date(text: str):
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if "time" in text and "date" not in text:
        return f"⏰ Time: {now.strftime('%I:%M %p')} (IST)"

    if "date" in text:
        return f"📅 Date: {now.strftime('%d %B %Y')}"

    return None

# ------------------------------------------------------
# GEMINI RESPONSE
# ------------------------------------------------------
def gemini_reply(question: str, context: str = ""):
    prompt = f"""
You are an AI assistant for Lovely Professional University (LPU).

STRICT RULES:
- If context is provided, answer ONLY from it
- Do NOT assume or invent LPU-specific rules
- Be precise, factual, and concise
- Do NOT generate dates or numbers unless present in context

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    try:
        client = get_gemini_client()
        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return res.text.strip()
    except Exception:
        logging.exception("Gemini error")
        return "Sorry, I couldn’t process that right now."

# ------------------------------------------------------
# CORE MESSAGE PROCESSOR
# ------------------------------------------------------
def process_message(msg: str) -> str:
    text = msg.lower().strip()

    # 1. Greeting
    greeting = handle_greeting(text)
    if greeting:
        return greeting

    # 2. Time / Date
    time_reply = handle_time_date(text)
    if time_reply:
        return time_reply

    # 3. Bot identity
    if any(k in text for k in ["who developed you", "who created you", "your developer"]):
        return (
            "I was developed by Sujith Lavudu and Vennela Barnana "
            "for Lovely Professional University."
        )

    # 4. LPU-first logic
    LPU_TERMS = [
        "lpu", "lovely professional university",
        "ums", "rms", "dsw",
        "attendance", "hostel", "fees",
        "exam", "semester", "registration",
        "reappear", "mid term", "end term"
    ]

    if any(k in text for k in LPU_TERMS):
        admin_answer = search_admin_content(msg)
        if admin_answer.strip():
            return gemini_reply(msg, admin_answer)

        knowledge_answer = search_lpu_knowledge(msg, STATIC_LPU)
        if knowledge_answer.strip():
            return gemini_reply(msg, knowledge_answer)

        return gemini_reply(
            msg,
            "No verified LPU-specific information is available for this question."
        )

    # 5. General fallback
    return gemini_reply(msg)

# ------------------------------------------------------
# CHAT API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Please enter a valid question."}

    return {"reply": process_message(message)}