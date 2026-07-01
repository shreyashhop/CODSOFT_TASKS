"""
Rule-Based Chatbot
===================
A simple chatbot that uses pattern matching (regex) and if-else logic
to recognize user intents and generate appropriate responses.

This demonstrates the fundamentals of rule-based NLP:
- Pattern matching against user input
- Intent recognition via keywords/regex
- Simple slot extraction (e.g., pulling a name out of a sentence)
- A fallback mechanism for unrecognized input
- Basic conversation state (remembering the user's name)

Run it with:  python rule_based_chatbot.py
"""

import random
import re
from datetime import datetime


class RuleBasedChatbot:
    def __init__(self, bot_name="RuleBot"):
        self.bot_name = bot_name
        self.user_name = None
        # exit_words trigger the end of the conversation
        self.exit_words = {"bye", "exit", "quit", "goodbye", "see you"}

        # Each rule is a (regex_pattern, handler_function) pair.
        # We check them in order, top to bottom, and use the first match.
        # re.IGNORECASE makes matching case-insensitive.
        self.rules = [
            (r"\b(hi|hello|hey|good morning|good evening)\b", self._handle_greeting),
            (r"\bmy name is (\w+)", self._handle_introduce_regex),
            (r"\bi am (\w+)\b", self._handle_introduce_regex),
            (r"\bwhat('?s| is) your name\b", self._handle_bot_name),
            (r"\bhow are you\b", self._handle_how_are_you),
            (r"\bwhat can you do\b|\bhelp\b", self._handle_help),
            (r"\b(time|what time is it)\b", self._handle_time),
            (r"\b(date|what.*today.*date|what day is it)\b", self._handle_date),
            (r"\bthank(s| you)\b", self._handle_thanks),
            (r"\b(joke|make me laugh)\b", self._handle_joke),
            (r"\bweather\b", self._handle_weather),
            (r"\bwho (created|made) you\b", self._handle_creator),
            (r"\b(bye|exit|quit|goodbye|see you)\b", self._handle_exit),
        ]

        self.fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Hmm, I don't have a rule for that yet. Try asking something else!",
            "Sorry, I didn't quite catch that. Can you say it differently?",
            "I'm just a simple rule-based bot — could you try simpler phrasing?",
        ]

    # ---------- Individual rule handlers ----------
    # Each handler receives the regex match object (may be unused) and
    # returns the text response.

    def _handle_greeting(self, match):
        greeting = f"Hello{', ' + self.user_name if self.user_name else ''}! How can I help you today?"
        return greeting

    def _handle_introduce_regex(self, match):
        name = match.group(1).capitalize()
        self.user_name = name
        return f"Nice to meet you, {name}! I'll remember your name for this chat."

    def _handle_bot_name(self, match):
        return f"I'm {self.bot_name}, your friendly rule-based chatbot."

    def _handle_how_are_you(self, match):
        return "I'm just a program, so I don't have feelings, but I'm running smoothly! How about you?"

    def _handle_help(self, match):
        return (
            "I can chat about a few things! Try asking me:\n"
            "  - 'What's your name?'\n"
            "  - 'What time is it?'\n"
            "  - 'Tell me a joke'\n"
            "  - 'My name is <yourname>'\n"
            "  - 'bye' to end our chat"
        )

    def _handle_time(self, match):
        now = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {now}."

    def _handle_date(self, match):
        today = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {today}."

    def _handle_thanks(self, match):
        return "You're welcome!"

    def _handle_joke(self, match):
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the chatbot go to therapy? It had too many unresolved queries.",
            "I told my computer I needed a break, and it said 'No problem, I'll go to sleep.'",
        ]
        return random.choice(jokes)

    def _handle_weather(self, match):
        return "I can't check live weather (I'm rule-based, not connected to the internet), but I hope it's nice outside!"

    def _handle_creator(self, match):
        return "I was built as a simple demonstration of rule-based chatbot design."

    def _handle_exit(self, match):
        name_part = f", {self.user_name}" if self.user_name else ""
        return f"Goodbye{name_part}! Have a great day. __EXIT__"

    # ---------- Core matching logic ----------

    def get_response(self, user_input):
        """Check user_input against each rule in order; return first match's response."""
        text = user_input.strip()
        if not text:
            return "Please say something!"

        for pattern, handler in self.rules:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return handler(match)

        # No rule matched -> fallback
        return random.choice(self.fallback_responses)

    def chat(self):
        """Run an interactive command-line chat loop."""
        print(f"{self.bot_name}: Hi! I'm {self.bot_name}. Type 'bye' to end our chat.\n")
        while True:
            user_input = input("You: ")
            response = self.get_response(user_input)

            if response.endswith("__EXIT__"):
                print(f"{self.bot_name}: {response.replace(' __EXIT__', '')}")
                break

            print(f"{self.bot_name}: {response}")


if __name__ == "__main__":
    bot = RuleBasedChatbot(bot_name="RuleBot")
    bot.chat()
