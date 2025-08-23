import os
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from uuid import uuid4
import re

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
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
CORS(app, resources={
    r"/chat": {"origins": "*"}, 
    r"/nutrition": {"origins": "*"},
    r"/analyze_symptoms": {"origins": "*"},
    r"/drug_interaction": {"origins": "*"},
    r"/voice_chat": {"origins": "*"},
    r"/export_chat": {"origins": "*"}
})

# Configure session
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid4()))
app.permanent_session_lifetime = timedelta(hours=4)

# ---------------------------
# Groq client setup
# ---------------------------
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama3-70b-8192"
MAX_CONVERSATION_HISTORY = 100

# ---------------------------
# Enhanced System Prompts
# ---------------------------
SYSTEM_PROMPT_CHAT = (
    "You are Nannu, an advanced AI medical assistant with expertise in general medicine, "
    "diagnostics, and patient care. When responding:\n"
    "• Always use clear bullet points (*) and numbered lists\n"
    "• Separate main ideas with paragraph breaks\n"
    "• For urgent symptoms, clearly state 'URGENT: Seek immediate medical attention'\n"
    "• Provide severity assessment (Low/Medium/High concern)\n"
    "• Ask relevant follow-up questions\n"
    "• Include helpful health tips when appropriate\n"
    "• Always end with: 'This is AI guidance, not a professional diagnosis. Consult a healthcare provider.'\n"
    "Be empathetic, thorough, and prioritize patient safety."
)

SYSTEM_PROMPT_SYMPTOMS = (
    "You are a medical symptom analyzer. Analyze the provided symptoms and provide:\n"
    "1. **Possible Conditions** (most likely to least likely)\n"
    "2. **Severity Assessment** (Low/Medium/High/Emergency)\n"
    "3. **Recommended Actions**\n"
    "4. **When to Seek Care** (timeline)\n"
    "5. **Warning Signs** to watch for\n"
    "Format with clear headers and bullet points. Always prioritize safety."
)

SYSTEM_PROMPT_NUTRITION = (
    "You are NutriBot, an advanced nutrition AI. Create detailed, personalized nutrition plans with:\n"
    "• **Daily Meal Plans** (breakfast, lunch, dinner, snacks)\n"
    "• **Macronutrient Breakdown** (carbs, protein, fats)\n"
    "• **Calorie Targets** based on goals\n"
    "• **Supplement Recommendations**\n"
    "• **Hydration Guidelines**\n"
    "• **Shopping List**\n"
    "Always include: 'Consult a registered dietitian for personalized advice.'"
)

SYSTEM_PROMPT_DRUG = (
    "You are a pharmaceutical interaction analyzer. For the provided medications:\n"
    "1. **Interaction Risk Level** (None/Low/Medium/High/Dangerous)\n"
    "2. **Specific Interactions** (detailed explanations)\n"
    "3. **Timing Recommendations**\n"
    "4. **Food Interactions**\n"
    "5. **Monitoring Advice**\n"
    "Always emphasize consulting a pharmacist or doctor."
)

# ---------------------------
# Utility Functions
# ---------------------------
def sanitize_input(text):
    """Sanitize user input to prevent XSS and other issues"""
    if not text:
        return ""
    # Remove potentially harmful characters
    text = re.sub(r'[<>"\']', '', text)
    # Limit length
    return text[:2000]

def format_ai_response(text):
    """Ensure AI response has proper formatting"""
    if not text:
        return text
    
    # Ensure bullet points are properly formatted
    if not any(marker in text for marker in ['*', '-', '•', '\n1.', '\n2.']):
        sentences = text.split('. ')
        if len(sentences) > 1:
            text = '\n• ' + '\n• '.join(sentence.strip() + '.' for sentence in sentences if sentence.strip())
    
    return text

def get_conversation_summary():
    """Get a summary of the current conversation"""
    if "conversation_history" not in session:
        return "No conversation yet."
    
    messages = session["conversation_history"][1:]  # Skip system prompt
    if not messages:
        return "No conversation yet."
    
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    return f"Topics discussed: {', '.join(user_messages[-3:])}"  # Last 3 topics

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
    session["chat_metadata"] = {
        "start_time": datetime.now().isoformat(),
        "message_count": 0,
        "topics": []
    }
    logger.info("🆕 Enhanced session started")
    return jsonify({"ok": True, "message": "New enhanced session started"})

@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured. Contact admin."}), 500
    
    if "conversation_history" not in session:
        session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    
    user_input = sanitize_input((request.json or {}).get("message", "").strip())
    chat_mode = (request.json or {}).get("mode", "normal")  # normal, detailed, quick
    
    if not user_input:
        return jsonify({"ok": False, "reply": "Please type a message."}), 400
    
    try:
        # Update metadata
        if "chat_metadata" not in session:
            session["chat_metadata"] = {"start_time": datetime.now().isoformat(), "message_count": 0, "topics": []}
        
        session["chat_metadata"]["message_count"] += 1
        
        # Adjust system prompt based on mode
        system_prompt = SYSTEM_PROMPT_CHAT
        if chat_mode == "detailed":
            system_prompt += "\n\nProvide detailed, comprehensive responses with additional medical context."
        elif chat_mode == "quick":
            system_prompt += "\n\nProvide concise, direct responses focusing on key points."
        
        # Add user input
        session["conversation_history"].append({"role": "user", "content": user_input})
        
        # Limit history
        if len(session["conversation_history"]) > MAX_CONVERSATION_HISTORY + 1:
            session["conversation_history"] = [session["conversation_history"][0]] + session["conversation_history"][-MAX_CONVERSATION_HISTORY:]
        
        # Determine temperature based on mode
        temperature = 0.3 if chat_mode == "detailed" else 0.5 if chat_mode == "quick" else 0.4
        
        # Call Groq
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=800 if chat_mode == "detailed" else 400 if chat_mode == "quick" else 600,
            messages=session["conversation_history"]
        )
        
        reply = resp.choices[0].message.content.strip()
        reply = format_ai_response(reply)
        
        session["conversation_history"].append({"role": "assistant", "content": reply})
        session.modified = True
        
        # Generate response metadata
        response_metadata = {
            "mode": chat_mode,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid4())[:8]
        }
        
        logger.info(f"✅ Chat response sent (mode: {chat_mode})")
        return jsonify({
            "ok": True, 
            "reply": reply, 
            "metadata": response_metadata,
            "conversation_summary": get_conversation_summary()
        })
        
    except Exception as e:
        logger.error(f"🔥 Groq error in /chat: {str(e)}")
        return jsonify({"ok": False, "reply": "Sorry, server error. Try again later."}), 500

@app.route("/analyze_symptoms", methods=["POST"])
def analyze_symptoms():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500
    
    data = request.json or {}
    symptoms = sanitize_input(data.get("symptoms", ""))
    duration = sanitize_input(data.get("duration", ""))
    severity = data.get("severity", "medium")
    age = data.get("age", "")
    gender = data.get("gender", "")
    
    if not symptoms:
        return jsonify({"ok": False, "reply": "Please describe your symptoms."}), 400
    
    try:
        prompt = f"""
        Analyze these symptoms for a {age}-year-old {gender}:
        
        Symptoms: {symptoms}
        Duration: {duration}
        Patient-reported severity: {severity}
        
        Provide a comprehensive analysis.
        """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SYMPTOMS},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,  # Lower temperature for medical analysis
            max_tokens=800,
            messages=messages
        )
        
        reply = resp.choices[0].message.content.strip()
        reply = format_ai_response(reply)
        
        logger.info("✅ Symptom analysis completed")
        return jsonify({
            "ok": True, 
            "reply": reply,
            "analysis_type": "symptoms",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔥 Error in symptom analysis: {str(e)}")
        return jsonify({"ok": False, "reply": "Error analyzing symptoms."}), 500

@app.route("/drug_interaction", methods=["POST"])
def drug_interaction():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500
    
    data = request.json or {}
    medications = sanitize_input(data.get("medications", ""))
    
    if not medications:
        return jsonify({"ok": False, "reply": "Please list your medications."}), 400
    
    try:
        prompt = f"""
        Check for drug interactions between these medications:
        {medications}
        
        Provide detailed interaction analysis and safety recommendations.
        """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DRUG},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,  # Very low temperature for drug interactions
            max_tokens=700,
            messages=messages
        )
        
        reply = resp.choices[0].message.content.strip()
        reply = format_ai_response(reply)
        
        logger.info("✅ Drug interaction analysis completed")
        return jsonify({
            "ok": True, 
            "reply": reply,
            "analysis_type": "drug_interaction",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔥 Error in drug interaction analysis: {str(e)}")
        return jsonify({"ok": False, "reply": "Error analyzing drug interactions."}), 500

@app.route("/nutrition", methods=["POST"])
def nutrition():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500
    
    data = request.json or {}
    required = ["age", "weight", "height", "goal", "duration"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"ok": False, "reply": f"Missing: {', '.join(missing)}"}), 400
    
    try:
        age, weight, height = float(data["age"]), float(data["weight"]), float(data["height"])
        activity_level = data.get("activity_level", "moderate")
        dietary_restrictions = sanitize_input(data.get("dietary_restrictions", "none"))
        medical_conditions = sanitize_input(data.get("medical_conditions", "none"))
        
        prompt = f"""
        Create a comprehensive nutrition plan for:
        - Age: {age} years
        - Weight: {weight}kg, Height: {height}cm
        - Goal: {data['goal']} over {data['duration']}
        - Activity Level: {activity_level}
        - Dietary Restrictions: {dietary_restrictions}
        - Medical Conditions: {medical_conditions}
        
        Include detailed meal plans, recipes, and shopping lists.
        """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NUTRITION},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            max_tokens=1000,
            messages=messages
        )
        
        reply = resp.choices[0].message.content.strip()
        reply = format_ai_response(reply)
        
        logger.info("✅ Enhanced nutrition plan generated")
        return jsonify({
            "ok": True, 
            "reply": reply,
            "plan_type": "nutrition",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔥 Error in nutrition planning: {str(e)}")
        return jsonify({"ok": False, "reply": "Error generating nutrition plan."}), 500

@app.route("/export_chat", methods=["POST"])
def export_chat():
    if "conversation_history" not in session:
        return jsonify({"ok": False, "message": "No conversation to export."}), 400
    
    try:
        # Prepare chat export
        chat_data = {
            "export_date": datetime.now().isoformat(),
            "session_metadata": session.get("chat_metadata", {}),
            "conversation": []
        }
        
        # Process conversation history (skip system prompt)
        for msg in session["conversation_history"][1:]:
            if msg["role"] in ["user", "assistant"]:
                chat_data["conversation"].append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.now().isoformat()  # In real app, store actual timestamps
                })
        
        return jsonify({
            "ok": True, 
            "chat_data": chat_data,
            "total_messages": len(chat_data["conversation"])
        })
        
    except Exception as e:
        logger.error(f"🔥 Error exporting chat: {str(e)}")
        return jsonify({"ok": False, "message": "Error exporting chat."}), 500

@app.route("/chat_statistics", methods=["GET"])
def chat_statistics():
    if "conversation_history" not in session:
        return jsonify({"ok": False, "stats": None})
    
    try:
        history = session["conversation_history"][1:]  # Skip system prompt
        user_messages = [msg for msg in history if msg["role"] == "user"]
        bot_messages = [msg for msg in history if msg["role"] == "assistant"]
        
        stats = {
            "total_messages": len(history),
            "user_messages": len(user_messages),
            "bot_messages": len(bot_messages),
            "session_duration": "Active",  # Could calculate actual duration
            "topics_discussed": len(set(msg["content"][:50] for msg in user_messages[-10:]))  # Rough topic count
        }
        
        return jsonify({"ok": True, "stats": stats})
        
    except Exception as e:
        logger.error(f"🔥 Error getting statistics: {str(e)}")
        return jsonify({"ok": False, "stats": None})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    session["chat_metadata"] = {
        "start_time": datetime.now().isoformat(),
        "message_count": 0,
        "topics": []
    }
    session.modified = True
    logger.info("🗑️ Enhanced history cleared")
    return jsonify({"ok": True, "message": "Conversation history cleared"})

# ---------------------------
# Error Handlers
# ---------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"ok": False, "message": "Internal server error"}), 500

# ---------------------------
# Run (local only)
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
