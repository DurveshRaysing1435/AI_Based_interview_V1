import sounddevice as sd
import soundfile as sf
import whisper
import librosa
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CHUNK_DURATION = 5  
SAMPLE_RATE = 16000 
TEMP_FILE = "live_chunk.wav" 

def init_models():
    print("\n[⏳] Initializing Core AI Engine...")
    print("[1/2] Loading Whisper (Speech-to-Text)...")
    whisper_model = whisper.load_model("base")
    print("[2/2] Loading Librosa (Risk Analysis)... Ready!")
    return whisper_model

def record_chunk():
    print(f"\n[🎙️] Listening for {CHUNK_DURATION} seconds... (Speak now!)")
    audio_data = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait() 
    sf.write(TEMP_FILE, audio_data, SAMPLE_RATE)

def analyze_chunk(whisper_model):
    # 1. WHISPER: Transcription
    result = whisper_model.transcribe(TEMP_FILE)
    transcript = result["text"].strip()

    # 2. LIBROSA: Risk Patterns & Behavior
    y, sr = librosa.load(TEMP_FILE, sr=None)
    
    # Calculate Volume (Max Amplitude)
    volume = np.max(np.abs(y)) if len(y) > 0 else 0
    
    # If the volume is above 0.02, someone is actually speaking
    if volume > 0.02:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        risk_data = f"{np.round(mfcc_mean[:4], 2)}... [Data Captured]"
        
        # Calculate Zero-Crossing Rate (Stress/Micro-tremor proxy)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # --- BEHAVIORAL ANALYSIS LOGIC ---
        if volume > 0.6:
            behavior_flag = "[⚠️ RAISED VOICE DETECTED]"
        elif zcr > 0.12:  # High frequency micro-tremors
            behavior_flag = "[⚠️ STRESS / NERVOUSNESS DETECTED]"
        else:
            behavior_flag = "[✅ NORMAL CALM VOICE]"
            
        if not transcript:
            transcript = "[Unclear Audio]"
            
    else:
        # It's too quiet, mark as Silence
        risk_data = "[No Audio Signal]"
        transcript = "[Silence]"
        behavior_flag = "[⏸️ SILENCE]"

    # --- LIVE DASHBOARD OUTPUT ---
    print("\n" + "="*65)
    print(" 🔴 LIVE INTERVIEW MONITORING DASHBOARD")
    print("="*65)
    print(f"🗣️  Speech: {transcript}")
    print(f"📉  Voice Risk: {risk_data}")
    print(f"🧠  Behavior: {behavior_flag}")
    print("="*65)

def main():
    try:
        w_model = init_models()
        print("\n✅ System Ready. Starting Live Monitoring...")
        print("Press Ctrl+C to stop.\n")
        
        while True:
            record_chunk()
            analyze_chunk(w_model)
            
    except KeyboardInterrupt:
        print("\n🛑 Live monitoring stopped by user.")
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()