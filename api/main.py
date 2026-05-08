from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.models.database import init_db
from api.routes import auth, search
from api.routes import auth, search, notifications

app = FastAPI(
    title="Visual Memory Lane API",
    description="Backend API for the Visual Memory Lane system",
    version="2.0.0"
)

# Allow requests from Streamlit, React, and React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PostgreSQL tables on startup
@app.on_event("startup")
def startup():
    init_db()

# Register routes
app.include_router(auth.router)
app.include_router(search.router)
app.include_router(notifications.router)

@app.get("/")
def root():
    return {"status": "Visual Memory Lane API running"}

@app.get("/health")
def health():
    return {"status": "ok"}