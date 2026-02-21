from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz
import re

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
# LOAD STATIC LPU KNOWLEDGE
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
# PERSON CONTEXT
# ------------------------------------------------------
PERSON_CONTEXT = {
    "sujith": """
Sujith Lavudu is a student innovator and software developer at Lovely Professional University.
He is the co-creator of the LPU VertoSewa AI assistant and co-author of the book 'Decode the Code'.
""",
    "vennela": """
Vennela Barnana is a researcher and author associated with Lovely Professional University.
She is the co-creator of the LPU VertoSewa AI assistant and co-author of the book 'Decode the Code'.
"""
}

# ------------------------------------------------------
# BASIC PROMPT INJECTION FILTER
# ------------------------------------------------------
def is_malicious(text: str) -> bool:
    patterns = [
        "ignore previous",
        "system prompt",
        "developer message",
        "reveal instructions",
        "act as"
    ]
    text = text.lower()
    return any(p in text for p in patterns)

# ------------------------------------------------------
# STATIC KNOWLEDGE SEARCH (Improved Scoring)
# ------------------------------------------------------
def search_static_knowledge(question: str):
    q_words = set(question.lower().split())
    chunks = STATIC_LPU.split("\n\n")
    scored = []

    for chunk in chunks:
        chunk_l = chunk.lower()
        score = sum(1 for w in q_words if w in chunk_l)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([c[1] for c in scored[:3]])

# ------------------------------------------------------
# ADMIN FIRESTORE SEARCH
# ------------------------------------------------------
def search_admin_content(question: str):
    db = get_db()
    q = question.lower()
    results = []

    docs = (
        db.collection("lpu_content")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(100)
        .stream()
    )

    for doc in docs:
        d = doc.to_dict()
        text_content = d.get("textContent", "")
        keywords = [k.lower() for k in (d.get("keywords") or [])]

        score = 0
        for k in keywords:
            if k in q:
                score += 2
        if any(word in text_content.lower() for word in q.split()):
            score += 1

        if score > 0:
            results.append((score, text_content))

    results.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([r[1] for r in results[:2]])

# ------------------------------------------------------
# GEMINI RESPONSE (STRICT GROUNDING FOR LPU)
# ------------------------------------------------------
def gemini_reply(question: str, context: str = "", strict_lpu=False):

    if strict_lpu:
        instruction = """
You are the official AI assistant of Lovely Professional University.

RULES:
- Use ONLY the provided CONTEXT to answer.
- If answer is not clearly available in the context, say:
  "Please contact the university administration for accurate information."
- Do not add external knowledge.
- Give a clean, clear, professional answer.
"""
    else:
        instruction = """
You are a helpful, intelligent assistant.
Give a clear, accurate and well-structured answer.
Do not mention context or internal reasoning.
"""

    prompt = f"""
{instruction}

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        logging.exception("Gemini error")
        return "Sorry, something went wrong."

# ------------------------------------------------------
# LPU DETECTION
# ------------------------------------------------------
LPU_TERMS = [
    "lpu", "lovely professional university",
    "ums", "rms", "dsw",
    "attendance", "hostel", "fees",
    "exam", "semester", "registration",
    "reappear", "mid term", "end term",
    "student organization", "soc",
    "mooc", "nptel", "swayam"
]

def is_lpu_related(text: str):
    text = text.lower()
    return any(term in text for term in LPU_TERMS)

# ------------------------------------------------------
# CORE MESSAGE PROCESSOR
# ------------------------------------------------------
def process_message(message: str):

    text = message.lower().strip()

    # Injection protection
    if is_malicious(text):
        return "I cannot process that request."

    # Greeting
    if text.startswith(("hi", "hello", "hey", "hai", "namaste")):
        return "Hello 👋 I’m LPU VertoSewa. How can I assist you today?"

    # Time / Date
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if "time" in text and "date" not in text:
        return f"⏰ {now.strftime('%I:%M %p')} (IST)"

    if "date" in text:
        return f"📅 {now.strftime('%d %B %Y')}"

    # Person context
    for name, context in PERSON_CONTEXT.items():
        if name in text:
            return gemini_reply(message, context)

    # LPU Questions (STRICT GROUNDED)
    if is_lpu_related(text):

        static_answer = search_static_knowledge(message)
        admin_answer = search_admin_content(message)

        combined_context = f"{static_answer}\n\n{admin_answer}"

        return gemini_reply(
            question=message,
            context=combined_context,
            strict_lpu=True
        )

    # General Questions
    return gemini_reply(message)

# ------------------------------------------------------
# CHAT ENDPOINT
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):

    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Please enter a valid question."}

    reply = process_message(message)
    return {"reply": reply}
