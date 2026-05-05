from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.models.database import get_db, ObjectLog, LLMLog, IntervalSummary, ActionLog
from api.models.schemas import SearchQuery
from clip_search import search_frames, embed_all_frames
from typing import List

router = APIRouter(prefix="/search", tags=["search"])

@router.get("/objects")
def search_objects(query: str, db: Session = Depends(get_db)):
    results = db.query(ObjectLog).filter(
        ObjectLog.object_name.ilike(f"%{query}%")
    ).order_by(ObjectLog.last_seen.desc()).limit(20).all()
    return results

@router.post("/visual")
def visual_search(body: SearchQuery):
    results = search_frames(body.query, top_k=body.top_k)
    return [{"frame_path": r[0], "score": round(r[1], 4)} for r in results]

@router.post("/index-frames")
def index_frames():
    embed_all_frames()
    return {"status": "indexed"}

@router.get("/summaries")
def get_summaries(limit: int = 20, db: Session = Depends(get_db)):
    results = db.query(IntervalSummary).order_by(
        IntervalSummary.interval_start.desc()
    ).limit(limit).all()
    return results

@router.get("/logs")
def get_logs(limit: int = 20, db: Session = Depends(get_db)):
    results = db.query(ActionLog).order_by(
        ActionLog.timestamp.desc()
    ).limit(limit).all()
    return results