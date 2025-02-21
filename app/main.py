from fastapi import FastAPI
from app.routes import chat

app = FastAPI()

app.include_router(chat.router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Mental Therapy Chatbot API is running."}
