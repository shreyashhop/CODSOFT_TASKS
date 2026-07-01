# Rule-Based Chatbot

A simple chatbot built in Python that responds to user input using **pattern matching (regex)** and **if-else style rule logic** — no machine learning required. It's a great starting point for understanding the basics of natural language processing and conversation flow.

## Features

- Regex-based intent recognition (matches keywords/phrases regardless of exact wording)
- Remembers the user's name during the conversation
- Handles greetings, small talk, jokes, date/time queries, and more
- Graceful fallback responses when input isn't understood
- Clean exit on words like `bye`, `exit`, `quit`, `goodbye`

## Requirements

- Python 3.6+
- No external libraries needed (uses only the standard library: `re`, `random`, `datetime`)

## How to Run

```bash
python rule_based_chatbot.py
```

Then just start typing. Example session:

```
RuleBot: Hi! I'm RuleBot. Type 'bye' to end our chat.

You: hello
RuleBot: Hello! How can I help you today?

You: my name is Alex
RuleBot: Nice to meet you, Alex! I'll remember your name for this chat.

You: what time is it
RuleBot: The current time is 14:32:10.

You: tell me a joke
RuleBot: Why do programmers prefer dark mode? Because light attracts bugs!

You: bye
RuleBot: Goodbye, Alex! Have a great day.
```

## Supported Commands / Intents

| Try saying...                          | Bot will...                              |
|----------------------------------------|-------------------------------------------|
| `hi`, `hello`, `hey`                   | Greet you                                  |
| `my name is <name>` / `I am <name>`    | Remember your name                         |
| `what's your name?`                    | Tell you its name                          |
| `how are you?`                         | Respond with a status                      |
| `what can you do?` / `help`            | List available commands                    |
| `what time is it?`                     | Give the current time                      |
| `what's the date?`                     | Give today's date                          |
| `thanks` / `thank you`                 | Respond politely                           |
| `tell me a joke`                       | Tell a random joke                         |
| `weather`                              | Explain it can't fetch live weather        |
| `who made you?`                        | Explain its purpose                        |
| `bye` / `exit` / `quit` / `goodbye`    | End the conversation                       |
| *(anything unrecognized)*              | Give a random fallback response            |

## How It Works

The chatbot is built around a list of **rules**, where each rule pairs a regex pattern with a handler function:

```python
self.rules = [
    (r"\b(hi|hello|hey)\b", self._handle_greeting),
    (r"\bmy name is (\w+)", self._handle_introduce_regex),
    ...
]
```

For every message, the bot checks each pattern **in order** and calls the handler for the **first match**. If nothing matches, it falls back to a randomly chosen "I don't understand" response. This mirrors a real (simplified) rule-based NLP pipeline:

1. **Pattern matching** — regex identifies the likely intent
2. **Slot extraction** — e.g., capturing the name from "my name is Alex"
3. **State tracking** — remembering the name for later use
4. **Response generation** — returning the appropriate reply
5. **Fallback handling** — catching anything the rules don't cover

## Extending the Bot

Want to add more capabilities? Just add a new tuple to `self.rules`:

```python
(r"\bfavorite color\b", self._handle_favorite_color),
```

...and define the matching handler method. You can also:

- Add more slots to extract (e.g., location, age, preferences)
- Add multi-turn context (remembering the last topic discussed)
- Swap in a real NLP library (like `nltk` or `spaCy`) once you outgrow simple rules
- Wrap it in a web interface using Flask or FastAPI

## File Structure

```
rule_based_chatbot.py   # Main chatbot script — rules, handlers, and chat loop
README.md                # This file
```

## License

Free to use and modify for learning purposes.
