from pyannote.audio import Pipeline

HF_TOKEN = "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"

print("⏳ Attempting to connect to Hugging Face and download Pyannote...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    print("\n✅ SUCCESS! Pyannote downloaded and loaded perfectly.")
    print("You can now run your main app.py server!")
except Exception as e:
    print(f"\n❌ FAILED. The network or token is still blocking it.\nError Details: {e}")