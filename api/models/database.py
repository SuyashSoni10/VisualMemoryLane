from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Table definitions ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.now)

class ObjectLog(Base):
    __tablename__ = "object_log"
    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String, index=True)
    first_seen = Column(String)
    last_seen = Column(String)
    duration_seconds = Column(Integer)
    status = Column(String)
    user_id = Column(Integer, nullable=True)

class LLMLog(Base):
    __tablename__ = "llm_log"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    scene_description = Column(Text)
    suggestion = Column(Text)
    user_id = Column(Integer, nullable=True)

class IntervalSummary(Base):
    __tablename__ = "interval_summary"
    id = Column(Integer, primary_key=True, index=True)
    interval_start = Column(String)
    interval_end = Column(String)
    summary = Column(Text)
    user_id = Column(Integer, nullable=True)

class ActionLog(Base):
    __tablename__ = "action_log"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    action_type = Column(String)
    detail = Column(Text)
    user_id = Column(Integer, nullable=True)

class PushToken(Base):
    __tablename__ = "push_tokens"
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()