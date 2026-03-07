import whisper
import warnings
import os

# Suppress minor warnings for a cleaner output in the terminal
warnings.filterwarnings("ignore")

def test_transcription(audio_path):
    print("[⏳] Loading Whisper 'base' model...")
    
    # 1. Verify the file exists using the exact path
    if not os.path.exists(audio_path):
        print(f"\n❌ Error: Python cannot find the file at: {audio_path}")
        return

    try:
        # 2. Load the base model.
        model = whisper.load_model("base")
        
        print(f"\n[🎤] Transcribing '{audio_path}'...")
        # 3. Transcribe the audio
        result = model.transcribe(audio_path)
        
        # 4. Print the final output
        print("\n" + "="*50)
        print(" ✅ TRANSCRIPTION SUCCESSFUL")
        print("="*50)
        print(f"Text: {result['text'].strip()}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ An error occurred during transcription: {e}")

if __name__ == "__main__":
    # Updated to the exact filename Windows created
    sample_file = r"C:\Users\durve\AI_Interview_Monitor\src\test_audio.m4a.m4a" 
    
    test_transcription(sample_file)