from flask import Flask, render_template, request, jsonify
import whisper
import librosa
import numpy as np
from sklearn.cluster import KMeans
import warnings
import os

warnings.filterwarnings("ignore")
app = Flask(__name__)

# --- INITIALIZE OFFLINE AI ENGINE ---
print("\n[⏳] Starting Fully Offline AI Engine...")
print("[1/3] Loading Whisper (Transcription)...")
whisper_model = whisper.load_model("base")

print("[2/3] Loading Librosa (Risk Analysis)... Ready!")

print("[3/3] Loading Offline ML Clustering (Speaker Detection)... Ready!")

print("\n🚀 FLASK SERVER LIVE! Go to http://127.0.0.1:5000\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio"})
    
    file = request.files['audio_data']
    temp_path = "web_chunk.webm"
    file.save(temp_path)

    transcript = "[Silence]"
    vocal_state = "Silent 😶"
    multi_speaker_warning = "Safe (1 Candidate)"

    try:
        # MODULE 1: WHISPER (The Ears)
        result = whisper_model.transcribe(temp_path)
        if result["text"].strip():
            transcript = result["text"].strip()
            
        print(f"🗣️ Server Heard: {transcript}")

        # Extract Audio Data once for Modules 2 & 3
        y, sr = librosa.load(temp_path, sr=None)
        
        if len(y) > 0 and np.max(np.abs(y)) > 0.02:
            vol = np.max(np.abs(y))
            
            # MODULE 2: LIBROSA (Vocal Emotion)
            if vol < 0.15:
                vocal_state = "Calm 😌"
            elif vol <= 0.40:
                vocal_state = "Stressed 😰"
            else:
                vocal_state = "Raised Voice 😡"

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = mfccs.T # Transpose for Machine Learning
            
            # MODULE 3: OFFLINE SPEAKER DETECTION (K-Means Clustering)
            # We ask the AI to split the voice data into 2 groups (clusters)
            if len(features) > 10:
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)
                
                # Count how much time each "voice pattern" spoke
                total_frames = len(kmeans.labels_)
                voice_1_ratio = np.count_nonzero(kmeans.labels_ == 0) / total_frames
                voice_2_ratio = np.count_nonzero(kmeans.labels_ == 1) / total_frames
                
                # If a second distinct voice takes up more than 20% of the audio chunk, flag it!
                if voice_1_ratio > 0.20 and voice_2_ratio > 0.20:
                    multi_speaker_warning = "🚨 RISK: 2 Voices Detected! (ML Cluster)"
                else:
                    multi_speaker_warning = "Safe (1 Candidate)"
        else:
            vocal_state = "Silent 😶"
            multi_speaker_warning = "Safe (Silence)"

    except Exception as e:
        print(f"Processing error: {e}")

    return jsonify({
        "transcript": transcript,
        "vocal_state": vocal_state,
        "speaker_warning": multi_speaker_warning
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)