import os
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from uuid import uuid4
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app, resources={r"/chat": {"origins": "*"}, r"/nutrition": {"origins": "*"}})

# Configure session
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid4()))
app.permanent_session_lifetime = timedelta(hours=2)  # Session expires after 2 hours

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# System prompts
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

MODEL_NAME = "llama3-70b-8192"

# Maximum conversation history to keep (in messages)
MAX_CONVERSATION_HISTORY = 50

@app.route("/")
def index():
    """Render the main application page."""
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    """Initialize a new session with empty conversation history."""
    session.permanent = True
    session['conversation_history'] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    logger.info("New session started")
    return jsonify({"ok": True, "message": "New session started"})

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests with symptom-related queries."""
    if client is None:
        logger.error("Groq client not configured: Missing GROQ_API_KEY")
        return jsonify({"ok": False, "reply": "Server configuration error. Please try again later."}), 500

    # Initialize conversation history if it doesn't exist
    if 'conversation_history' not in session:
        session['conversation_history'] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]

    user_input = (request.json or {}).get("message", "").strip()
    if not user_input:
        logger.warning("Empty message received in /chat")
        return jsonify({"ok": False, "reply": "Please type a message."}), 400
    if len(user_input) > 500:  # Basic input length validation
        logger.warning("Message too long in /chat")
        return jsonify({"ok": False, "reply": "Message too long. Please keep it under 500 characters."}), 400

    try:
        # Add user message to conversation history
        session['conversation_history'].append({"role": "user", "content": user_input})
        
        # Ensure we don't exceed maximum history length (keep system prompt)
        if len(session['conversation_history']) > MAX_CONVERSATION_HISTORY + 1:  # +1 for system prompt
            # Keep system prompt and the most recent messages
            session['conversation_history'] = (
                [session['conversation_history'][0]] + 
                session['conversation_history'][-MAX_CONVERSATION_HISTORY:]
            )

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            max_tokens=600,
            messages=session['conversation_history']
        )
        
        reply = resp.choices[0].message.content.strip()
        
        # Add assistant response to conversation history
        session['conversation_history'].append({"role": "assistant", "content": reply})
        session.modified = True  # Ensure Flask knows the session was modified
        
        logger.info("Successful chat response generated")
        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"ok": False, "reply": "Sorry, an error occurred. Please try again."}), 500

@app.route("/nutrition", methods=["POST"])
def nutrition():
    """Generate a personalized nutrition plan based on user data."""
    if client is None:
        logger.error("Groq client not configured: Missing GROQ_API_KEY")
        return jsonify({"ok": False, "reply": "Server configuration error. Please try again later."}), 500

    data = request.json or {}
    required_fields = ['age', 'weight', 'height', 'goal', 'duration']
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        logger.warning(f"Missing fields in /nutrition: {missing_fields}")
        return jsonify({"ok": False, "reply": f"Please provide {', '.join(missing_fields)}."}), 400

    try:
        # Validate numeric inputs
        age = float(data['age'])
        weight = float(data['weight'])
        height = float(data['height'])
        if age < 1 or age > 120:
            return jsonify({"ok": False, "reply": "Age must be between 1 and 120."}), 400
        if weight < 1 or weight > 500:
            return jsonify({"ok": False, "reply": "Weight must be between 1 and 500 kg."}), 400
        if height < 1 or height > 300:
            return jsonify({"ok": False, "reply": "Height must be between 1 and 300 cm."}), 400

        # Validate categorical inputs
        valid_goals = ['lose weight', 'gain weight', 'maintain weight']
        valid_durations = ['1 week', '2 weeks', '1 month']
        if data['goal'] not in valid_goals:
            return jsonify({"ok": False, "reply": "Invalid goal selected."}), 400
        if data['duration'] not in valid_durations:
            return jsonify({"ok": False, "reply": "Invalid duration selected."}), 400

        prompt = (
            f"Generate a nutrition plan for a {data['age']}-year-old, weighing {data['weight']}kg, "
            f"{data['height']}cm tall, with a goal to {data['goal']} over {data['duration']}. "
            f"Include daily meal suggestions, calorie targets, macronutrient balance, and hydration tips."
        )

        # Use fresh conversation for nutrition to avoid mixing with chat history
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
        logger.info("Successful nutrition plan generated")
        return jsonify({"ok": True, "reply": reply})
    except ValueError:
        logger.warning("Invalid numeric input in /nutrition")
        return jsonify({"ok": False, "reply": "Please provide valid numeric values for age, weight, and height."}), 400
    except Exception as e:
        logger.error(f"Error in /nutrition: {str(e)}")
        return jsonify({"ok": False, "reply": "Sorry, an error occurred. Please try again."}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear the conversation history."""
    if 'conversation_history' in session:
        session['conversation_history'] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
        session.modified = True
        logger.info("Conversation history cleared")
    return jsonify({"ok": True, "message": "Conversation history cleared"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)