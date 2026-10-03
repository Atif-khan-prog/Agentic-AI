from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_pipline import get_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "https://ragpipeline.netlify.app"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG API is running!"}


@app.post("/chat")
def chat(request: ChatRequest):
    answer = get_answer(request.question)

    return {
        "question": request.question,
        "answer": answer
    }