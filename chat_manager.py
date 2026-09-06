import json
import os
import uuid
from datetime import datetime


CHAT_FILE = "data/chats.json"


class ChatManager:

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.chats = {}
        self.load()

    def load(self):

        if not os.path.exists(CHAT_FILE):
            self.chats = {}
            return

        try:
            with open(
                CHAT_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if isinstance(data, dict):
                    self.chats = data
                else:
                    self.chats = {}

        except Exception as e:

            print("Chat loading error:", e)
            self.chats = {}

    def save(self):

        os.makedirs("data", exist_ok=True)

        with open(
            CHAT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.chats,
                file,
                indent=4,
                ensure_ascii=False
            )

    def create_chat(self, title="New Chat"):

        chat_id = str(uuid.uuid4())

        self.chats[chat_id] = {
            "id": chat_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

        self.save()

        return chat_id

    def chat_exists(self, chat_id):

        return chat_id in self.chats

    def get_chat(self, chat_id):

        return self.chats.get(chat_id)

    def get_all_chats(self):

        return list(self.chats.values())

    def add_message(
        self,
        chat_id,
        role,
        content
    ):

        if chat_id not in self.chats:
            return False

        self.chats[chat_id]["messages"].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
        )

        self.save()

        return True

    def get_messages(self, chat_id):

        chat = self.get_chat(chat_id)

        if not chat:
            return []

        return chat.get("messages", [])

    def get_message_count(self, chat_id):

        return len(
            self.get_messages(chat_id)
        )

    def get_context(
        self,
        chat_id,
        max_messages=20
    ):

        messages = self.get_messages(chat_id)

        if not messages:
            return "No previous conversation."

        messages = messages[-max_messages:]

        context = []

        for message in messages:

            role = message.get(
                "role",
                "unknown"
            ).upper()

            content = message.get(
                "content",
                ""
            )

            context.append(
                f"{role}: {content}"
            )

        return "\n".join(context)

    def delete_chat(self, chat_id):

        if chat_id not in self.chats:
            return False

        del self.chats[chat_id]

        self.save()

        return True

    def clear_chat(self, chat_id):

        if chat_id not in self.chats:
            return False

        self.chats[chat_id]["messages"] = []

        self.save()

        return True

    def update_title(
        self,
        chat_id,
        title
    ):

        if chat_id not in self.chats:
            return False

        self.chats[chat_id]["title"] = title

        self.save()

        return True

    def generate_title(
        self,
        chat_id,
        first_message
    ):

        if chat_id not in self.chats:
            return False

        title = first_message.strip()

        if not title:
            title = "New Chat"

        if len(title) > 35:
            title = title[:35] + "..."

        self.chats[chat_id]["title"] = title

        self.save()

        return True

    def clear_all(self):

        self.chats = {}

        self.save()


if __name__ == "__main__":

    print("========================================")
    print("       CHAT MANAGER TEST")
    print("========================================")

    manager = ChatManager()

    chat_id = manager.create_chat(
        "Test Chat"
    )

    manager.add_message(
        chat_id,
        "user",
        "Hello"
    )

    manager.add_message(
        chat_id,
        "assistant",
        "Hello! How can I help?"
    )

    print("\nChat ID:")
    print(chat_id)

    print("\nMessages:")
    print(manager.get_messages(chat_id))

    print("\nContext:")
    print(manager.get_context(chat_id))

    print("\nMessage Count:")
    print(manager.get_message_count(chat_id))

    print("\nChatManager working successfully!")