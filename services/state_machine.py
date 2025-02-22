from typing import List


class SimpleStateMachine:
    def __init__(self):
        self.history: List[dict] = []

    def update(self, user_msg: str, bot_msg: str):
        self.history.append({"user": user_msg, "bot": bot_msg})

    def get_history(self) -> str:
        conversation = ""
        for turn in self.history:
            conversation += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        return conversation


# Global instance for simplicity; consider per-session storage in production.
state_machine = SimpleStateMachine()
