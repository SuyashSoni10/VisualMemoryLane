import pyttsx3
import threading

def speak(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[VOICE] Error: {e}")

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()