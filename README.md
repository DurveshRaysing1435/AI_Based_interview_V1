## 🚀 Project Overview
A real-time web application designed to monitor candidates during technical interviews. It uses three specialized AI modules to ensure interview integrity and provide live vocal analytics without requiring continuous cloud API dependencies.

## 📂 Project Structure
* `app.py`: Main Flask server.
* `requirements.txt`: Python dependencies list.
* `test_pyannote.py`: Network testing script.
* `templates/`: Web interface folder.
* `index.html`: Web dashboard UI.
* `src/`: Local development scripts.

## 🛠️ Integrated AI Modules
1. **Module 1: Speech-to-Text (OpenAI Whisper)** - Live candidate transcription.
2. **Module 2: Vocal Emotion & Risk (Librosa)** - Detects stress/calm states.
3. **Module 3: Speaker Detection (Offline ML)** - K-Means multi-speaker detection.

## 💻 How to Run the Module
1. **Open terminal** in your project directory.
2. **Create virtual environment:** `python -m venv venv`
3. **Activate environment:** * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Start AI server:** `python app.py`
6. **Access dashboard:** Open your web browser and navigate to `http://127.0.0.1:5000`
