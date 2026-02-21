from fastapi import FastAPI, Request
import os
import logging
from datetime import datetime
import pytz

from google.cloud import firestore
from google import genai

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
        return ""

STATIC_LPU = load_lpu_knowledge()

# ------------------------------------------------------
# LPU TERMS
# ------------------------------------------------------
LPU_TERMS = [
    "lpu", "lovely professional university",
    "ums", "rms", "dsw",
    "attendance", "hostel", "fees",
    "exam", "semester", "registration",
    "reappear", "mid term", "end term",
    "student organization", "soc",
    "mooc", "nptel", "swayam",
    "ncc", "pro chancellor", "chancellor"
]

def is_lpu_related(text: str):
    return any(term in text.lower() for term in LPU_TERMS)

# ------------------------------------------------------
# SEARCH STATIC
# ------------------------------------------------------
def search_static(question: str):
    q_words = question.lower().split()
    chunks = STATIC_LPU.split("\n\n")
    scored = []

    for chunk in chunks:
        score = sum(1 for w in q_words if w in chunk.lower())
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join(c[1] for c in scored[:3])

# ------------------------------------------------------
# SEARCH ADMIN
# ------------------------------------------------------
def search_admin(question: str):
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
        data = doc.to_dict()
        content = data.get("textContent", "")
        if any(word in content.lower() for word in q.split()):
            results.append(content)

    return "\n\n".join(results[:2])

# ------------------------------------------------------
# GEMINI
# ------------------------------------------------------
def gemini_reply(question: str, context: str = ""):

    if context:
        instruction = """
You are the official AI assistant of Lovely Professional University.

Use the CONTEXT below to answer accurately.
If additional clarification is needed, answer clearly.
Do not mention the context.
"""
    else:
        instruction = """
You are a helpful assistant.
Provide a clear, structured, accurate answer.
Do not mention internal reasoning.
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
        return "Sorry, something went wrong."

# ------------------------------------------------------
# PROCESS MESSAGE
# ------------------------------------------------------
def process_message(message: str):

    text = message.lower().strip()

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

    # LPU FLOW
    if is_lpu_related(text):

        static_data = search_static(message)
        admin_data = search_admin(message)

        combined_context = f"{static_data}\n\n{admin_data}".strip()

        # ✅ If context found → grounded
        if combined_context:
            return gemini_reply(message, combined_context)

        # ✅ If no context → fallback to Gemini general knowledge
        return gemini_reply(message)

    # General questions
    return gemini_reply(message)

# ------------------------------------------------------
# API
# ------------------------------------------------------
@app.post("/chat")
async def chat_api(request: Request):

    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Please enter a valid question."}

    reply = process_message(message)
    return {"reply": reply}
