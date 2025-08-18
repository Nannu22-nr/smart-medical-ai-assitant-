import os
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from uuid import uuid4
from datetime import timedelta

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()  # Works locally (ignored in Render)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("❌ Missing GROQ_API_KEY. Please set it in Render Environment Variables.")
else:
    logger.info("✅ GROQ_API_KEY detected")

# ---------------------------
# Flask app setup
# ---------------------------
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app, resources={r"/chat": {"origins": "*"}, r"/nutrition": {"origins": "*"}})

# Configure session
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid4()))
app.permanent_session_lifetime = timedelta(hours=2)

# ---------------------------
# Groq client setup
# ---------------------------
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama3-70b-8192"
MAX_CONVERSATION_HISTORY = 50

# ---------------------------
# System Prompts
# ---------------------------
SYSTEM_PROMPT_CHAT = (
    "You are Ohidul Alam Nannu, a careful, friendly medical AI assistant. "
    "Provide evidence-informed guidance for symptom triage, common conditions, "
    "self-care tips, and red-flag advisories. Be concise (5–8 short sentences). "
    "ALWAYS add a short disclaimer: 'This is general guidance, not a diagnosis.' "
    "If symptoms sound urgent, clearly recommend local emergency care. "
    "Ask 2–4 focused follow-up questions when useful. Keep the tone warm and clear."
)

SYSTEM_PROMPT_NUTRITION = (
    "You are NutriGuide, a helpful nutrition assistant. Based on provided user data (age, weight, height, goal, duration), "
    "generate a detailed, personalized weekly nutrition plan. Include daily meal suggestions, approximate calorie targets, "
    "macronutrient balance, and hydration tips. Ensure the plan is practical, culturally adaptable, and safe. "
    "ALWAYS add: 'Consult a registered dietitian for personalized advice.'"
)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    session.permanent = True
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    logger.info("🆕 New session started")
    return jsonify({"ok": True, "message": "New session started"})

@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured. Contact admin."}), 500

    if "conversation_history" not in session:
        session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]

    user_input = (request.json or {}).get("message", "").strip()
    if not user_input:
        return jsonify({"ok": False, "reply": "Please type a message."}), 400

    try:
        # Add user input
        session["conversation_history"].append({"role": "user", "content": user_input})

        # Limit history
        if len(session["conversation_history"]) > MAX_CONVERSATION_HISTORY + 1:
            session["conversation_history"] = [session["conversation_history"][0]] + session["conversation_history"][-MAX_CONVERSATION_HISTORY:]

        # Call Groq
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            max_tokens=600,
            messages=session["conversation_history"]
        )

        reply = resp.choices[0].message.content.strip()
        session["conversation_history"].append({"role": "assistant", "content": reply})
        session.modified = True

        logger.info("✅ Chat response sent")
        return jsonify({"ok": True, "reply": reply})

    except Exception as e:
        logger.error(f"🔥 Groq error in /chat: {str(e)}")
        return jsonify({"ok": False, "reply": "Sorry, server error. Try again later."}), 500

@app.route("/nutrition", methods=["POST"])
def nutrition():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured. Contact admin."}), 500

    data = request.json or {}
    required = ["age", "weight", "height", "goal", "duration"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"ok": False, "reply": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        age, weight, height = float(data["age"]), float(data["weight"]), float(data["height"])

        prompt = (
            f"Generate a nutrition plan for a {age}-year-old, weighing {weight}kg, "
            f"{height}cm tall, with a goal to {data['goal']} over {data['duration']}."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NUTRITION},
            {"role": "user", "content": prompt}
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.5,
            max_tokens=700,
            messages=messages
        )

        reply = resp.choices[0].message.content.strip()
        logger.info("✅ Nutrition plan generated")
        return jsonify({"ok": True, "reply": reply})

    except Exception as e:
        logger.error(f"🔥 Groq error in /nutrition: {str(e)}")
        return jsonify({"ok": False, "reply": "Server error while generating plan."}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    session.modified = True
    logger.info("🗑️ History cleared")
    return jsonify({"ok": True, "message": "Conversation history cleared"})

# ---------------------------
# Run (local only)
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
