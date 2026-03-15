from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from rag_pipeline import ask_question, initialize_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag()
    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Medical RAG API running"}


@app.post("/ask")
def ask(query: Query):

    answer = ask_question(query.question)

    return {
        "question": query.question,
        "answer": answer
    }
# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

