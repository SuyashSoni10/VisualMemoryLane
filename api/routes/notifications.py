from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import httpx
from api.models.database import get_db, PushToken

router = APIRouter(prefix="/notifications", tags=["notifications"])

class TokenRequest(BaseModel):
    token: str

class PushMessage(BaseModel):
    title: str
    body: str

async def send_push_notification(token: str, title: str, body: str):
    message = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": {"type": "alert"}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://exp.host/--/api/v2/push/send",
            json=message,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
    return response.json()

async def notify_all_devices(title: str, body: str, db: Session):
    tokens = db.query(PushToken).all()
    for t in tokens:
        try:
            await send_push_notification(t.token, title, body)
        except Exception as e:
            print(f"[PUSH] Failed to send to {t.token}: {e}")

@router.post("/register-token")
def register_token(request: TokenRequest, db: Session = Depends(get_db)):
    existing = db.query(PushToken).filter(
        PushToken.token == request.token
    ).first()
    if not existing:
        db.add(PushToken(token=request.token))
        db.commit()
    return {"status": "token registered"}

@router.post("/test")
async def test_notification(db: Session = Depends(get_db)):
    await notify_all_devices(
        "Test Notification",
        "Visual Memory Lane is working!",
        db
    )
    return {"status": "sent"}

@router.post("/send")
async def send_notification(msg: PushMessage, db: Session = Depends(get_db)):
    await notify_all_devices(msg.title, msg.body, db)
    return {"status": "sent"}