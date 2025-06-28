from flask import Flask, request, render_template, redirect, url_for, send_file
import os
from gtts import gTTS
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

from ai.emotion import analyze_emotion

emotion = analyze_emotion(user_data)
if emotion == "frustrated":
    suggestion = "You seem stuck — want help?"
elif emotion == "confident":
    suggestion = "You're doing amazing — ready for a challenge?"

from ai.memory import load_memory, save_memory
from ai.conversation import generate_response
from ai.predictor import predict_next_action

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    user_id = "user123"  # future: replace with real session/user id
    memory = load_memory()
    user_data = memory["users"].get(user_id, {
        "gesture_history": {},
        "topics_learned": [],
        "voice_prefs": {},
        "recent_conversations": []
    })

    output = ""
    suggestion = ""

    if request.method == "POST":
        gesture = request.form.get("gesture")
        output = gestures.get(gesture, "Gesture not recognized.")
        response = generate_response(output)
        # Score gesture consistency
        user_data.setdefault("gesture_score", {})
        user_data["gesture_score"][gesture] = user_data["gesture_score"].get(gesture, 0) + 1

        # Unlock "learning moments" after 3 repeats
        if user_data["gesture_score"][gesture] == 3:
            suggestion = f"You're getting great with '{gesture}' — want to try something new?"

        # Log gesture usage
        user_data["gesture_history"][gesture] = user_data["gesture_history"].get(gesture, 0) + 1

        # Add conversation memory
        user_data["recent_conversations"].append({
            "input": gesture,
            "response": response
        })

        suggestion = predict_next_action(user_data)
        speak_text(response)

    memory["users"][user_id] = user_data
    save_memory(memory)

    return render_template("index.html", output=output, suggestion=suggestion)

app = Flask(__name__, template_folder="templates")
print(f"Template folder path: {app.template_folder}")  # Debug print

# A simple dictionary to store gestures and their meanings
gestures = dict(
    blink_once="Hello",
    blink_twice="Yes",
    look_left="No",
    hand_wave="Goodbye",
    head_tilt="I need help"
)

# Function to convert text to speech and play it
def speak_text(text: str) -> None:
    try:
        tts = gTTS(text=text, lang='en')
        audio_path = os.path.join(os.path.dirname(__file__), "temp_output.mp3")
        tts.save(audio_path)
        os.system(f"mpg123 {audio_path}")  # Use mpg123 for Linux (Replit)
    except Exception as e:
        print(f"Speech error: {e}")

# Redirect root to the start page
@app.route("/")
def root():
    return redirect(url_for('start'))

# Welcome page to start a session
@app.route("/start", methods=["GET", "POST"])
def start():
    if request.method == "POST":
        name = request.form.get("name")
        return redirect(url_for('home', name=name))
    return render_template("start.html")

# Home page where users enter gestures
@app.route("/home", methods=["GET", "POST"])
def home():
    output = ""
    if request.method == "POST":
        gesture = request.form.get("gesture")
        output = gestures.get(gesture, "Gesture not recognized.")  # type: ignore
        speak_text(output)  # Speak the output
    return render_template("index.html", output=output)

# Page to add new gestures
@app.route("/customize", methods=["POST"])
def customize():
    new_gesture = request.form.get("new_gesture")
    meaning = request.form.get("meaning")
    gestures[new_gesture] = meaning  # Add the new gesture
    output = f"Added: {new_gesture} -> {meaning}"
    speak_text(output)  # Speak the output
    return render_template("index.html", output=output)

# Add a route to download the audio file
@app.route("/download_audio")
def download_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "temp_output.mp3")
    return send_file(audio_path, as_attachment=True, download_name="output.mp3")
