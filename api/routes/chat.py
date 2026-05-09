from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from groq import Groq
from api.models.database import get_db, ObjectLog, IntervalSummary, LLMLog
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/chat", tags=["chat"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHAT_SYSTEM_PROMPT = """You are a smart visual memory assistant. You have access to 
data about objects detected in a person's workspace over time. Answer the user's 
questions based on this data naturally and helpfully. Be concise and specific. 
If the data doesn't contain enough information to answer, say so honestly."""

class ChatRequest(BaseModel):
    message: str
    category: str = "Personal"

def build_context(db: Session) -> str:
    # Get last 20 object logs
    objects = db.query(ObjectLog).order_by(
        ObjectLog.last_seen.desc()
    ).limit(20).all()

    # Get last 5 summaries
    summaries = db.query(IntervalSummary).order_by(
        IntervalSummary.interval_start.desc()
    ).limit(5).all()

    # Get last 5 LLM suggestions
    suggestions = db.query(LLMLog).order_by(
        LLMLog.timestamp.desc()
    ).limit(5).all()

    context = "=== WORKSPACE MEMORY ===\n\n"

    if objects:
        context += "Recent objects detected:\n"
        for o in objects:
            mins = o.duration_seconds // 60 if o.duration_seconds else 0
            context += f"- {o.object_name} | {o.status} | {mins} mins | last seen: {o.last_seen}\n"

    if summaries:
        context += "\nRecent interval summaries:\n"
        for s in summaries:
            context += f"- [{s.interval_start} → {s.interval_end}]: {s.summary}\n"

    if suggestions:
        context += "\nRecent AI suggestions:\n"
        for s in suggestions:
            context += f"- [{s.timestamp}]: {s.suggestion}\n"

    return context

@router.post("/message")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    context = build_context(db)

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nUser question: {request.message}"}
            ]
        )
        response = completion.choices[0].message.content
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}