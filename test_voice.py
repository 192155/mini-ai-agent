import pyttsx3


print("Starting voice test...")


engine = pyttsx3.init()

engine.setProperty(
    "rate",
    170
)

engine.setProperty(
    "volume",
    1.0
)


text = "Hello Parit. Your Mini AI Agent voice output is working."


print("Speaking...")

engine.say(text)

engine.runAndWait()

print("Voice test completed.")