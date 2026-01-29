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
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "models/gemini-2.5-flash"

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
# SEARCH ADMIN CONTENT
# ------------------------------------------------------
def search_admin_content(question: str):
    db = get_db()
    q = question.lower()
    matches = []

    docs = (
        db.collection("lpu_content")
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(50)
        .stream()
    )

    for doc in docs:
        d = doc.to_dict()
        text = (d.get("textContent") or "").lower()
        keywords = d.get("keywords") or []
        category = (d.get("category") or "").lower()

        if any(k.lower() in q for k in keywords) or category in q:
            matches.append(f"{d.get('title','')}:\n{d.get('textContent','')}")

    return "\n\n".join(matches)

# ------------------------------------------------------
# GREETING
# ------------------------------------------------------
def handle_greeting(text: str):
    if text in ["hi", "hello", "hey", "hii", "hai", "namaste"]:
        return (
            Hello 👋 Welcome to LPU VertoSewa
An AI-powered assistant developed for Lovely Professional University (LPU).

I can help you with information related to:

• Academics – shedules, rules
• Hostels & Fees – policies, payments, queries
• RMS / UMS – registrations, portals, procedures
• DSW Notices – updates and announcements
• People & General Information
• Date & Time
        )
    return None

# ------------------------------------------------------
# TIME & DATE (STRICTLY PYTHON)
# ------------------------------------------------------
def handle_time_date(text: str):
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)

    if text in ["time", "time now"]:
        return f"⏰ Time: {now_ist.strftime('%I:%M %p')} (IST)"

    if text in ["date", "date today", "today date", "date only"]:
        return f"📅 Date: {now_ist.strftime('%d %B %Y')}"

    if "time in america" in text or "time in usa" in text:
        et = pytz.timezone("US/Eastern")
        ct = pytz.timezone("US/Central")
        pt = pytz.timezone("US/Pacific")
        return (
            "🇺🇸 United States Time:\n"
            f"• Eastern (ET): {datetime.now(et).strftime('%I:%M %p')}\n"
            f"• Central (CT): {datetime.now(ct).strftime('%I:%M %p')}\n"
            f"• Pacific (PT): {datetime.now(pt).strftime('%I:%M %p')}"
        )

    if "time in singapore" in text:
        sg = pytz.timezone("Asia/Singapore")
        return f"🇸🇬 Singapore Time: {datetime.now(sg).strftime('%I:%M %p')}"

    return None

# ------------------------------------------------------
# GEMINI RESPONSE
# ------------------------------------------------------
def gemini_reply(question: str, context: str = ""):
    prompt = f"""
You are an educational assistant.

Rules:
- Reply only in English
- Be accurate and clear
- Use provided context as VERIFIED facts
- You may add general, publicly known information
- Do NOT invent private, sensitive, or unverifiable details
- Do NOT generate dates or real-time info

CONTEXT:
{context}

QUESTION:
{question}
"""
    try:
        client = get_gemini_client()
        res = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return res.text.strip()
    except Exception as e:
        logging.error(e)
        return "Please try again later."

# ------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------
def process_message(msg: str) -> str:
    text = msg.lower().strip()

    # 1️⃣ TIME / DATE
    time_reply = handle_time_date(text)
    if time_reply:
        return time_reply

    # 2️⃣ PERSON QUERIES (FIXED + GEMINI)
    PERSON_CONTEXT = ""

    if "sujith lavudu" in text:
        PERSON_CONTEXT = """
Sujith Lavudu is a student innovator, software developer, and author.
He is the co-creator of the LPU Vertosewa AI Assistant and co-author of the book 'Decode the Code'.
"""

    elif "vennela barnana" in text:
        PERSON_CONTEXT = """
Vennela Barnana is an author and researcher.
She is the co-creator of the LPU Vertosewa AI Assistant and co-author of the book 'Decode the Code'.
"""

    elif "rashmi mittal" in text:
        PERSON_CONTEXT = """
Dr. Rashmi Mittal is the Pro-Chancellor of Lovely Professional University (LPU).
"""

    if PERSON_CONTEXT:
        return gemini_reply(msg, PERSON_CONTEXT)

    # 3️⃣ GREETING
    greet = handle_greeting(text)
    if greet:
        return greet

    # 4️⃣ BOT DEVELOPER IDENTITY
    if any(k in text for k in [
        "who developed you",
        "who created you",
        "who is the developer",
        "your developer",
        "your creator"
    ]):
        return (
            "I was developed by Sujith Lavudu and Vennela Barnana "
            "for Lovely Professional University (LPU)."
        )

    # 5️⃣ LPU-FIRST STRATEGY
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

        if STATIC_LPU.strip():
            return gemini_reply(msg, STATIC_LPU)

        # fallback → Gemini general knowledge (Google-known LPU info)
        return gemini_reply(msg)

    # 6️⃣ GENERAL QUESTIONS
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