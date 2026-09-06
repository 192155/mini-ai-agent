import json
import os


MEMORY_FILE = "data/chat_memory.json"


class ConversationMemory:

    def __init__(self, max_messages=20):

        self.max_messages = max_messages

        os.makedirs(
            "data",
            exist_ok=True
        )

        self.messages = []

        self.load()


    def add_message(self, role, content):

        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )

        # Sirf latest messages rakho
        if len(self.messages) > self.max_messages:

            self.messages = self.messages[
                -self.max_messages:
            ]

        self.save()


    def get_messages(self):

        return self.messages


    def get_context(self):

        if not self.messages:

            return "No previous conversation."

        context = []

        for message in self.messages:

            role = message["role"].upper()

            content = message["content"]

            context.append(
                f"{role}: {content}"
            )

        return "\n".join(context)


    def clear(self):

        self.messages = []

        self.save()


    def save(self):

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.messages,
                file,
                indent=4,
                ensure_ascii=False
            )


    def load(self):

        if not os.path.exists(
            MEMORY_FILE
        ):

            return

        try:

            with open(
                MEMORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                self.messages = json.load(
                    file
                )

        except Exception:

            self.messages = []


if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "      MEMORY SYSTEM TEST"
    )

    print(
        "=============================="
    )

    memory = ConversationMemory()

    memory.add_message(
        "user",
        "My name is Parit."
    )

    memory.add_message(
        "assistant",
        "Nice to meet you, Parit."
    )

    print(
        "\nSaved conversation:\n"
    )

    print(
        memory.get_context()
    )

    print(
        "\nMemory system working successfully."
    )