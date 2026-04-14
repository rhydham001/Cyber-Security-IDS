"""FastAPI application entry-point for the Intelligent IDS."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="Intelligent Cyber-Security IDS API",
    description="Intrusion Detection System powered by Random Forest & XGBoost on NSL-KDD",
    version="1.0.0",
)

# Allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Intelligent IDS API is running", "docs": "/docs"}
