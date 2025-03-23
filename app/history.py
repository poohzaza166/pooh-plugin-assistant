import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .utils import provide_data, register_with_message_bus, subscribe

#

class History:
    def __init__(self):
        self.conversationHistory = []
    
    @provide_data("conversation_history")
    def get_user_conversation(self):
        """Get the full conversation history"""
        return self.conversationHistory
    
    @provide_data("latest_conversation")
    def get_latest_covnersation(self):
        return self.conversationHistory[-1]

    @subscribe("text_input")
    def store_text_input(self, content, **kwargs):
        """Store user input in the conversation history"""
        self.conversationHistory.append({"content": content, "user": "user"})
        print(f"Stored message: {content}")

    

def register_history_thing(message):
    why = History()
    register_with_message_bus(why, message)
    return why
