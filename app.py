import os
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from uuid import uuid4
import re
import onnxruntime as ort
from PIL import Image
import io
import base64
import pandas as pd
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("❌ Missing GROQ_API_KEY. Please set it in Render Environment Variables.")
else:
    logger.info("✅ GROQ_API_KEY detected")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app, resources={
    r"/chat": {"origins": "*"}, 
    r"/nutrition": {"origins": "*"},
    r"/analyze_symptoms": {"origins": "*"},
    r"/drug_interaction": {"origins": "*"},
    r"/voice_chat": {"origins": "*"},
    r"/export_chat": {"origins": "*"},
    r"/brain_tumor_prediction": {"origins": "*"},
    r"/upload_health_record": {"origins": "*"},
    r"/predict_risk": {"origins": "*"}
})

app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid4()))
app.permanent_session_lifetime = timedelta(hours=4)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ONNX_MODEL_PATH = os.path.join("models", "mobilenetv2_classifier.onnx")
brain_tumor_model = None

try:
    if os.path.exists(ONNX_MODEL_PATH):
        brain_tumor_model = ort.InferenceSession(ONNX_MODEL_PATH)
        logger.info("✅ Brain tumor classifier model loaded successfully")
        input_details = brain_tumor_model.get_inputs()[0]
        output_details = brain_tumor_model.get_outputs()[0]
        logger.info(f"Model input shape: {input_details.shape}, type: {input_details.type}")
        logger.info(f"Model output shape: {output_details.shape}, type: {output_details.type}")
    else:
        logger.warning(f"⚠️ Brain tumor model not found at {ONNX_MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading brain tumor model: {str(e)}")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama3-70b-8192"
MAX_CONVERSATION_HISTORY = 100

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

SYSTEM_PROMPT_BRAIN_TUMOR = (
    "You are a medical AI specializing in brain tumor analysis. When interpreting brain scan results:\n"
    "1. **Explain the AI prediction results clearly**\n"
    "2. **Provide context about what the prediction means**\n"
    "3. **Emphasize the need for professional medical evaluation**\n"
    "4. **Suggest appropriate next steps**\n"
    "5. **Include relevant information about brain tumors**\n"
    "Always stress that AI predictions are supplementary tools and cannot replace professional medical diagnosis."
)

SYSTEM_PROMPT_HEALTH_RECORD = (
    "You are a health record analyzer. Analyze the provided health report data and provide:\n"
    "1. **Summary of Key Findings**\n"
    "2. **Identified Abnormalities** (with explanations)\n"
    "3. **Potential Health Concerns**\n"
    "4. **Suggested Follow-up Questions** for the patient or doctor\n"
    "5. **Recommendations**\n"
    "Format with clear headers and bullet points. Always prioritize safety and emphasize consulting a healthcare professional."
)

SYSTEM_PROMPT_RISK_TIPS = (
    "You are a health risk advisor. Based on the provided risk type, score, and patient details, provide:\n"
    "1. **Risk Interpretation**\n"
    "2. **Personalized Lifestyle Tips**\n"
    "3. **Preventive Measures**\n"
    "4. **When to Consult a Doctor**\n"
    "Format with clear headers and bullet points. Be encouraging and focus on actionable advice."
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_brain_image(image_data, target_size=(128, 128)):
    try:
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(image, dtype=np.float32)
        
        img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Preprocessed image shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_brain_tumor(image_data):
    if not brain_tumor_model:
        return {"error": "Brain tumor model not available"}
    
    try:
        processed_image = preprocess_brain_image(image_data)
        if processed_image is None:
            return {"error": "Failed to preprocess image"}
        
        input_name = brain_tumor_model.get_inputs()[0].name
        logger.info(f"Model input name: {input_name}")
        logger.info(f"Input shape being sent: {processed_image.shape}")
        
        outputs = brain_tumor_model.run(None, {input_name: processed_image})
        
        predictions = outputs[0][0]
        logger.info(f"Raw model output shape: {outputs[0].shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        class_names = [
            "No Tumor",
            "Glioma Tumor", 
            "Meningioma Tumor",
            "Pituitary Tumor"
        ]
        
        if len(predictions.shape) == 0:
            confidence = float(predictions)
            if confidence > 0.5:
                predicted_class = "Tumor Detected"
                confidence_percentage = confidence * 100
            else:
                predicted_class = "No Tumor"
                confidence_percentage = (1 - confidence) * 100
                
            return {
                "prediction": predicted_class,
                "confidence": round(confidence_percentage, 2),
                "raw_output": float(predictions),
                "model_info": {
                    "model_type": "Brain Tumor Classifier (MobileNetV2)",
                    "input_shape": processed_image.shape,
                    "classification_type": "binary"
                }
            }
            
        elif len(predictions) == len(class_names):
            if np.max(predictions) > 1.0 or np.min(predictions) < 0.0:
                exp_preds = np.exp(predictions - np.max(predictions))
                predictions = exp_preds / np.sum(exp_preds)
            
            results = []
            for i, class_name in enumerate(class_names):
                confidence = float(predictions[i])
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "percentage": round(confidence * 100, 2)
                })
            
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "prediction": results[0]["class"],
                "confidence": results[0]["percentage"],
                "all_results": results,
                "model_info": {
                    "model_type": "Brain Tumor Classifier (MobileNetV2)",
                    "input_shape": processed_image.shape,
                    "classes": len(class_names),
                    "classification_type": "multi-class"
                }
            }
        else:
            max_idx = np.argmax(predictions)
            confidence = float(predictions[max_idx])
            
            if np.max(predictions) > 1.0 or np.min(predictions) < 0.0:
                exp_preds = np.exp(predictions - np.max(predictions))
                softmax_preds = exp_preds / np.sum(exp_preds)
                confidence = float(softmax_preds[max_idx])
            
            predicted_class = class_names[max_idx] if max_idx < len(class_names) else f"Class_{max_idx}"
            
            return {
                "prediction": predicted_class,
                "confidence": round(confidence * 100, 2),
                "raw_output": predictions.tolist(),
                "model_info": {
                    "model_type": "Brain Tumor Classifier (MobileNetV2)",
                    "input_shape": processed_image.shape,
                    "output_classes": len(predictions)
                }
            }
            
    except Exception as e:
        logger.error(f"Error in brain tumor prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def sanitize_input(text):
    if not text:
        return ""
    text = re.sub(r'[<>"\']', '', text)
    return text[:2000]

def format_ai_response(text):
    if not text:
        return text
    
    if not any(marker in text for marker in ['*', '-', '•', '\n1.', '\n2.']):
        sentences = text.split('. ')
        if len(sentences) > 1:
            text = '\n• ' + '\n• '.join(sentence.strip() + '.' for sentence in sentences if sentence.strip())
    
    return text

def get_conversation_summary():
    if "conversation_history" not in session:
        return "No conversation yet."
    
    messages = session["conversation_history"][1:]
    if not messages:
        return "No conversation yet."
    
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    return f"Topics discussed: {', '.join(user_messages[-3:])}"

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
    chat_mode = (request.json or {}).get("mode", "normal")
    
    if not user_input:
        return jsonify({"ok": False, "reply": "Please type a message."}), 400
    
    try:
        if "chat_metadata" not in session:
            session["chat_metadata"] = {"start_time": datetime.now().isoformat(), "message_count": 0, "topics": []}
        
        session["chat_metadata"]["message_count"] += 1
        
        system_prompt = SYSTEM_PROMPT_CHAT
        if chat_mode == "detailed":
            system_prompt += "\n\nProvide detailed, comprehensive responses with additional medical context."
        elif chat_mode == "quick":
            system_prompt += "\n\nProvide concise, direct responses focusing on key points."
        
        session["conversation_history"].append({"role": "user", "content": user_input})
        
        if len(session["conversation_history"]) > MAX_CONVERSATION_HISTORY + 1:
            session["conversation_history"] = [session["conversation_history"][0]] + session["conversation_history"][-MAX_CONVERSATION_HISTORY:]
        
        temperature = 0.3 if chat_mode == "detailed" else 0.5 if chat_mode == "quick" else 0.4
        
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

@app.route("/brain_tumor_prediction", methods=["POST"])
def brain_tumor_prediction():
    if not brain_tumor_model:
        return jsonify({
            "ok": False, 
            "reply": "Brain tumor prediction model is not available. Please contact administrator."
        }), 503
    
    try:
        image_data = None
        
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                if allowed_file(file.filename):
                    image_data = file.read()
                else:
                    return jsonify({
                        "ok": False, 
                        "reply": "Invalid file format. Please upload PNG, JPG, JPEG, GIF, BMP, or TIFF files."
                    }), 400
        
        elif request.json and 'image_data' in request.json:
            try:
                image_data = base64.b64decode(request.json['image_data'])
            except Exception as e:
                return jsonify({
                    "ok": False, 
                    "reply": "Invalid base64 image data."
                }), 400
        
        if not image_data:
            return jsonify({
                "ok": False, 
                "reply": "No image provided. Please upload a brain scan image."
            }), 400
        
        prediction_result = predict_brain_tumor(image_data)
        
        if "error" in prediction_result:
            return jsonify({
                "ok": False, 
                "reply": f"Prediction error: {prediction_result['error']}"
            }), 500
        
        if client:
            try:
                interpretation_prompt = f"""
                Interpret these brain tumor prediction results:
                
                Prediction: {prediction_result['prediction']}
                Confidence: {prediction_result['confidence']}%
                
                Additional details: {json.dumps(prediction_result.get('all_results', []), indent=2)}
                
                Please provide:
                1. What this prediction means
                2. Important disclaimers about AI predictions
                3. Recommended next steps
                4. When to seek immediate medical attention
                5. General information about the predicted condition (if tumor detected)
                """
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_BRAIN_TUMOR},
                    {"role": "user", "content": interpretation_prompt}
                ]
                
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.2,
                    max_tokens=800,
                    messages=messages
                )
                
                ai_interpretation = resp.choices[0].message.content.strip()
                ai_interpretation = format_ai_response(ai_interpretation)
                
            except Exception as e:
                logger.error(f"Error generating AI interpretation: {str(e)}")
                ai_interpretation = None
        else:
            ai_interpretation = None
        
        response_data = {
            "ok": True,
            "prediction_results": prediction_result,
            "ai_interpretation": ai_interpretation,
            "analysis_type": "brain_tumor_prediction",
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This AI prediction is for informational purposes only and should not replace professional medical diagnosis. Please consult with a qualified healthcare provider for proper evaluation and treatment."
        }
        
        logger.info(f"✅ Brain tumor prediction completed: {prediction_result['prediction']} ({prediction_result['confidence']}%)")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"🔥 Error in brain tumor prediction endpoint: {str(e)}")
        return jsonify({
            "ok": False, 
            "reply": "An error occurred during brain tumor analysis. Please try again."
        }), 500

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
            temperature=0.2,
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
        return jupytext({"ok": False, "reply": "Please list your medications."}), 400
    
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
            temperature=0.1,
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

@app.route("/upload_health_record", methods=["POST"])
def upload_health_record():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500

    if 'file' not in request.files:
        return jsonify({"ok": False, "reply": "No file provided."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "reply": "No selected file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "reply": "Invalid file format. Please upload PDF or CSV."}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        report_data = ""
        file_ext = filename.rsplit('.', 1)[1].lower()

        if file_ext == 'pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                report_data += page.extract_text() + "\n"
        elif file_ext == 'csv':
            df = pd.read_csv(file_path)
            report_data = df.to_string(index=False)

        if not report_data.strip():
            return jsonify({"ok": False, "reply": "Empty or unreadable file."}), 400

        prompt = f"""
        Analyze this health report (blood test, prescription, etc.):
        
        Report Content:
        {report_data[:5000]}  # Limit to avoid token overflow
        
        Summarize abnormalities and suggest follow-up questions.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_HEALTH_RECORD},
            {"role": "user", "content": prompt}
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=800,
            messages=messages
        )

        analysis = resp.choices[0].message.content.strip()
        analysis = format_ai_response(analysis)

        # Clean up uploaded file
        os.remove(file_path)

        logger.info("✅ Health record analysis completed")
        return jsonify({
            "ok": True,
            "analysis": analysis,
            "analysis_type": "health_record",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"🔥 Error in health record upload/analysis: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"ok": False, "reply": "Error processing health record."}), 500

def calculate_diabetes_risk(data):
    # Simple FINDRISC-based score (0-26)
    score = 0
    age = data.get('age', 0)
    bmi = data.get('bmi', 0)
    waist = data.get('waist_circumference', 0)
    physical_activity = data.get('physical_activity', False)  # True if >=30min/day
    veggies = data.get('daily_veggies_fruit', False)  # True if daily
    hypertension_med = data.get('hypertension_med', False)
    high_blood_sugar = data.get('high_blood_sugar_history', False)
    family_diabetes = data.get('family_diabetes', False)

    if age >= 45 and age < 55: score += 2
    elif age >= 55 and age < 65: score += 3
    elif age >= 65: score += 4

    if bmi >= 25 and bmi < 30: score += 1
    elif bmi >= 30: score += 3

    if (data.get('gender', 'male') == 'male' and waist >= 94) or (data.get('gender', 'female') == 'female' and waist >= 80):
        score += 3
    if (data.get('gender', 'male') == 'male' and waist >= 102) or (data.get('gender', 'female') == 'female' and waist >= 88):
        score += 1  # Additional

    if not physical_activity: score += 2
    if not veggies: score += 1
    if hypertension_med: score += 2
    if high_blood_sugar: score += 5
    if family_diabetes: score += 5  # Immediate family
    elif family_diabetes == 'distant': score += 3

    if score < 7: risk_level = 'Low'
    elif score < 12: risk_level = 'Slightly Elevated'
    elif score < 15: risk_level = 'Moderate'
    elif score < 20: risk_level = 'High'
    else: risk_level = 'Very High'

    return {'score': score, 'risk_level': risk_level}

def calculate_heart_disease_risk(data):
    # Simple Framingham-inspired score
    score = 0
    age = data.get('age', 0)
    gender = data.get('gender', 'male')
    cholesterol = data.get('total_cholesterol', 0)
    hdl = data.get('hdl_cholesterol', 0)
    systolic_bp = data.get('systolic_bp', 0)
    smoker = data.get('smoker', False)
    diabetes = data.get('diabetes', False)
    treated_bp = data.get('treated_hypertension', False)

    # Age points (simplified)
    if gender == 'male':
        if age >= 45 and age < 55: score += 3
        elif age >= 55: score += 6
    else:
        if age >= 50 and age < 60: score += 4
        elif age >= 60: score += 7

    # Cholesterol
    if cholesterol >= 200 and cholesterol < 240: score += 1
    elif cholesterol >= 240: score += 2

    # HDL
    if hdl < 40: score += 2
    elif hdl >= 60: score -= 1

    # BP
    if systolic_bp >= 130 and systolic_bp < 140: score += 1
    elif systolic_bp >= 140: score += 2
    if treated_bp: score += 1

    if smoker: score += 2
    if diabetes: score += 2

    if score < 3: risk_level = 'Low'
    elif score < 6: risk_level = 'Moderate'
    else: risk_level = 'High'

    return {'score': score, 'risk_level': risk_level}

def calculate_kidney_disease_risk(data):
    # Simple CKD risk score
    score = 0
    age = data.get('age', 0)
    gender = data.get('gender', 'male')
    hypertension = data.get('hypertension', False)
    diabetes = data.get('diabetes', False)
    bmi = data.get('bmi', 0)
    smoker = data.get('smoker', False)
    anemia = data.get('anemia', False)
    proteinuria = data.get('proteinuria', False)

    if age >= 50: score += 2
    if age >= 70: score += 2

    if gender == 'female': score += 1

    if hypertension: score += 3
    if diabetes: score += 3
    if bmi >= 30: score += 2
    if smoker: score += 1
    if anemia: score += 2
    if proteinuria: score += 4

    if score < 3: risk_level = 'Low'
    elif score < 7: risk_level = 'Moderate'
    else: risk_level = 'High'

    return {'score': score, 'risk_level': risk_level}

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    if not client:
        return jsonify({"ok": False, "reply": "Server not configured."}), 500

    data = request.json or {}
    risk_type = data.get("type")
    if not risk_type:
        return jsonify({"ok": False, "reply": "Missing risk type (diabetes, heart, kidney)."}), 400

    try:
        if risk_type == "diabetes":
            risk_result = calculate_diabetes_risk(data)
        elif risk_type == "heart":
            risk_result = calculate_heart_disease_risk(data)
        elif risk_type == "kidney":
            risk_result = calculate_kidney_disease_risk(data)
        else:
            return jsonify({"ok": False, "reply": "Invalid risk type. Choose diabetes, heart, or kidney."}), 400

        prompt = f"""
        Provide lifestyle tips for {risk_type} risk:
        
        Risk Level: {risk_result['risk_level']}
        Score: {risk_result['score']}
        Patient Details: {json.dumps(data, indent=2)}
        
        Focus on preventive measures and tips.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RISK_TIPS},
            {"role": "user", "content": prompt}
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            max_tokens=600,
            messages=messages
        )

        tips = resp.choices[0].message.content.strip()
        tips = format_ai_response(tips)

        logger.info(f"✅ {risk_type.capitalize()} risk prediction completed: {risk_result['risk_level']}")
        return jsonify({
            "ok": True,
            "risk_result": risk_result,
            "lifestyle_tips": tips,
            "risk_type": risk_type,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This is a simplified risk assessment. Consult a healthcare professional for accurate evaluation."
        })

    except Exception as e:
        logger.error(f"🔥 Error in risk prediction: {str(e)}")
        return jsonify({"ok": False, "reply": "Error predicting risk."}), 500

@app.route("/export_chat", methods=["POST"])
def export_chat():
    if "conversation_history" not in session:
        return jsonify({"ok": False, "message": "No conversation to export."}), 400
    
    try:
        chat_data = {
            "export_date": datetime.now().isoformat(),
            "session_metadata": session.get("chat_metadata", {}),
            "conversation": []
        }
        
        for msg in session["conversation_history"][1:]:
            if msg["role"] in ["user", "assistant"]:
                chat_data["conversation"].append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.now().isoformat()
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
        history = session["conversation_history"][1:]
        user_messages = [msg for msg in history if msg["role"] == "user"]
        bot_messages = [msg for msg in history if msg["role"] == "assistant"]
        
        stats = {
            "total_messages": len(history),
            "user_messages": len(user_messages),
            "bot_messages": len(bot_messages),
            "session_duration": "Active",
            "topics_discussed": len(set(msg["content"][:50] for msg in user_messages[-10:]))
        }
        
        return jsonify({"ok": True, "stats": stats})
        
    except Exception as e:
        logger.error(f"🔥 Error getting statistics: {str(e)}")
        return jsonify({"ok": False, "stats": None})

@app.route("/model_status", methods=["GET"])
def model_status():
    status = {
        "groq_llm": client is not None,
        "brain_tumor_classifier": brain_tumor_model is not None,
        "model_path": ONNX_MODEL_PATH,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
    }
    
    if brain_tumor_model:
        try:
            inputs = brain_tumor_model.get_inputs()
            outputs = brain_tumor_model.get_outputs()
            status["model_details"] = {
                "input_name": inputs[0].name,
                "input_shape": inputs[0].shape,
                "input_type": str(inputs[0].type),
                "output_name": outputs[0].name,
                "output_shape": outputs[0].shape,
                "output_type": str(outputs[0].type)
            }
        except Exception as e:
            status["model_details"] = f"Error getting model details: {str(e)}"
    
    return jsonify(status)

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

@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"ok": False, "message": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"ok": False, "message": "File too large. Maximum size is 16MB."}), 413

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 