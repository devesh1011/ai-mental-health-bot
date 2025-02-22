import os
from config import GROQ_API_KEY
from .state_machine import state_machine
from .prompts import build_prompt
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Instantiate the ChatGroq model
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b-32768",
    temperature=0.7,
)


def get_bot_response(user_message: str) -> str:
    conversation_history = state_machine.get_history()
    prompt = build_prompt(user_message, conversation_history)
    # Create the chain using your prompt, language model, and output parser.
    chain = prompt | llm | StrOutputParser()
    bot_message = ""
    # Stream the response (here we accumulate chunks; in production you might stream this directly)
    for chunk in chain.stream(
        {"input": user_message, "conversation_history": conversation_history}
    ):
        bot_message += chunk
    state_machine.update(user_message, bot_message)
    return bot_message
