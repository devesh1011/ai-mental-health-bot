from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chatbot import get_bot_response

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    user_message = chat_request.message
    bot_message = get_bot_response(user_message)
    return ChatResponse(response=bot_message)
