from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ObjectLogSchema(BaseModel):
    object_name: str
    first_seen: str
    last_seen: str
    duration_seconds: int
    status: str

    class Config:
        from_attributes = True

class LLMLogSchema(BaseModel):
    timestamp: str
    scene_description: str
    suggestion: str

    class Config:
        from_attributes = True

class SummarySchema(BaseModel):
    interval_start: str
    interval_end: str
    summary: str

    class Config:
        from_attributes = True

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5