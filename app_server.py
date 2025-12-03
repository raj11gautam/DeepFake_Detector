import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import werkzeug.utils

# Static folder set to current directory to serve deepfake_detector.html directly
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
OUTPUT_FOLDER = 'generated_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# IMPORT AGENTS (Safe Import)
try:
    import deepfake_detection_agent as video_agent
    import audio_detection_agent as audio_agent
    import voice_cloning_agent as clone_agent
    print("[INFO] All agents imported successfully.")
except ImportError as e:
    print(f"[FATAL ERROR] Missing libraries or agents: {e}")
    print("Did you run 'pip install -r requirements.txt'?")

# LOAD MODELS 
print("Loading AI models...")
face_detector, video_classifier, audio_classifier = None, None, None
try:
    face_detector, video_classifier = video_agent.load_ai_models()
    audio_classifier = audio_agent.load_audio_model()
    print("---* Server startup complete. READY. *---")
except Exception as e:
    print(f"[Warning] Some models failed to load: {e}")

# ROUTES 

# 1. Serve Interface (Home Page)
@app.route('/')
def serve_index():
    # Automatically serves deepfake_detector.html when you open http://127.0.0.1:5000
    return send_from_directory('.', 'deepfake_detector.html')

# 2. Video Analysis
@app.route('/analyze-video', methods=['POST'])
def analyze_video_route():
    if 'video' not in request.files: return jsonify({"status": "Failed", "error": "No file provided"}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({"status": "Failed", "error": "No file name"}), 400

    filename = werkzeug.utils.secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    try:
        report = video_agent.analyze_video(temp_path, face_detector, video_classifier)
    except Exception as e:
        print(f"[Error] Video analysis failed: {e}")
        report = {"status": "Failed", "error": str(e)}
    
    try: os.remove(temp_path) 
    except: pass
    return jsonify(report)

# 3. Audio Analysis
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio_route():
    if 'audio' not in request.files: return jsonify({"status": "Failed", "error": "No file provided"}), 400
    file = request.files['audio']
    filename = werkzeug.utils.secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    try:
        report = audio_agent.analyze_audio(temp_path, audio_classifier)
    except Exception as e:
        print(f"[Error] Audio analysis failed: {e}")
        report = {"status": "Failed", "error": str(e)}

    try: os.remove(temp_path)
    except: pass
    return jsonify(report)

# 4. Voice Cloning
@app.route('/clone-voice', methods=['POST'])
def clone_voice_route():
    if 'audio' not in request.files: return jsonify({"status": "Failed", "error": "No file provided"}), 400
    file = request.files['audio']
    text = request.form.get('text', '')
    
    filename = werkzeug.utils.secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    try:
        result = clone_agent.clone_voice(temp_path, text, OUTPUT_FOLDER)
        if result['status'] == 'Success':
            result['audioUrl'] = f"http://127.0.0.1:5000/generated/{result['outputFile']}"
    except Exception as e:
        print(f"[Error] Cloning failed: {e}")
        result = {"status": "Failed", "error": str(e)}

    try: os.remove(temp_path)
    except: pass
    return jsonify(result)

# 5. Generated Files
@app.route('/generated/<path:filename>')
def serve_generated(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)