import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from api.models.database import (
    Base, ObjectLog, LLMLog, IntervalSummary,
    ActionLog, init_db
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize tables on import
init_db()

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

def log_object(object_name, first_seen, last_seen, duration_seconds, status):
    print(f"[STORAGE] Writing to DB: {object_name} | {status}")
    db = SessionLocal()
    try:
        entry = ObjectLog(
            object_name=object_name,
            first_seen=first_seen,
            last_seen=last_seen,
            duration_seconds=duration_seconds,
            status=status
        )
        db.add(entry)
        db.commit()
        print(f"[STORAGE] Committed successfully")
    except Exception as e:
        db.rollback()
        print(f"[STORAGE] log_object error: {e}")
    finally:
        db.close()

def log_llm(scene_description, suggestion):
    db = SessionLocal()
    try:
        entry = LLMLog(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            scene_description=scene_description,
            suggestion=suggestion
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"log_llm error: {e}")
    finally:
        db.close()

def log_summary(interval_start, interval_end, summary):
    db = SessionLocal()
    try:
        entry = IntervalSummary(
            interval_start=interval_start,
            interval_end=interval_end,
            summary=summary
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"log_summary error: {e}")
    finally:
        db.close()

def log_action(action_type, detail):
    db = SessionLocal()
    try:
        entry = ActionLog(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action_type=action_type,
            detail=detail
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"log_action error: {e}")
    finally:
        db.close()

def search_objects(query):
    db = SessionLocal()
    try:
        results = db.query(ObjectLog).filter(
            ObjectLog.object_name.ilike(f"%{query}%")
        ).order_by(ObjectLog.last_seen.desc()).limit(20).all()
        print(f"[SEARCH] Query: {query} | Results found: {len(results)}")
        return [
            (r.object_name, r.first_seen, r.last_seen, r.duration_seconds, r.status)
            for r in results
        ]
    except Exception as e:
        print(f"[SEARCH] search_objects error: {e}")
        return []
    finally:
        db.close()

def get_recent_logs(limit=20):
    db = SessionLocal()
    try:
        results = db.query(ActionLog).order_by(
            ActionLog.timestamp.desc()
        ).limit(limit).all()
        return [(r.timestamp, r.action_type, r.detail) for r in results]
    except Exception as e:
        print(f"get_recent_logs error: {e}")
        return []
    finally:
        db.close()

def get_latest_llm():
    db = SessionLocal()
    try:
        result = db.query(LLMLog).order_by(
            LLMLog.timestamp.desc()
        ).first()
        if result:
            return (result.timestamp, result.suggestion)
        return None
    except Exception as e:
        print(f"get_latest_llm error: {e}")
        return None
    finally:
        db.close()

def get_summaries(limit=20):
    db = SessionLocal()
    try:
        results = db.query(IntervalSummary).order_by(
            IntervalSummary.interval_start.desc()
        ).limit(limit).all()
        return [
            (r.interval_start, r.interval_end, r.summary)
            for r in results
        ]
    except Exception as e:
        print(f"get_summaries error: {e}")
        return []
    finally:
        db.close()