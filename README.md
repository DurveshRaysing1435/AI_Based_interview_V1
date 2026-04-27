<div align="center">

# 🎙️ AI Interview Integrity Monitor
### Real-Time Transcription · Vocal Emotion Analysis · Multi-Speaker Detection

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20ML-FF6B35?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

> **Three specialized AI modules working in parallel — transcribing speech, detecting vocal stress, and identifying multiple speakers in real time, without continuous cloud dependencies.**

</div>

---

## 📌 What This Does

Proctoring a technical interview goes beyond watching a video feed. This system listens intelligently:

- 📝 **Transcribes candidate speech** live using OpenAI Whisper — locally, with no per-request API cost
- 😰 **Detects vocal stress and calm states** using audio feature analysis via Librosa
- 👥 **Flags multiple speakers** in the audio stream using offline K-Means clustering — catching third-party assistance in real time

---

## 🧩 System Architecture

```
Candidate Browser (index.html)
        │
        │  Audio Stream (MediaRecorder API)
        ▼
┌────────────────────────────────────┐
│         Flask Server               │  ← app.py
└──────────────┬─────────────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Module 1 │ │ Module 2 │ │   Module 3   │
│          │ │          │ │              │
│ Whisper  │ │ Librosa  │ │  K-Means ML  │
│   STT    │ │ Emotion  │ │   Speaker    │
│          │ │  & Risk  │ │  Detection   │
└──────────┘ └──────────┘ └──────────────┘
        │           │              │
        └───────────┴──────────────┘
                    │
             Live Dashboard
              (index.html)
```

---

## 📂 Project Structure

```
├── app.py                  # Flask server — routes and module orchestration
├── requirements.txt        # Python dependencies
├── test_pyannote.py        # Network connectivity testing script
├── templates/
│   └── index.html          # Live monitoring dashboard UI
└── src/                    # Local development and utility scripts
```

---

## 🛠️ Integrated AI Modules

### Module 1 — Speech-to-Text (`OpenAI Whisper`)

Transcribes candidate audio in real time directly on the server — no per-request cloud API call required once the model is loaded.

| Property | Detail |
|---|---|
| Engine | OpenAI Whisper (local inference) |
| Output | Live rolling transcript |
| Cloud dependency | ❌ None after model download |

---

### Module 2 — Vocal Emotion & Risk (`Librosa`)

Analyses the acoustic features of the candidate's voice — pitch variance, tempo, and energy — to classify their vocal state.

| Signal | Interpretation |
|---|---|
| 🟢 Calm, steady pitch | Normal response state |
| 🟡 Elevated energy / fast tempo | Mild stress detected |
| 🔴 High pitch variance + tension | Risk flag raised |

---

### Module 3 — Speaker Detection (Offline K-Means ML)

Runs entirely offline. Clusters audio embeddings to detect when more than one distinct voice is present in the stream — a strong signal of third-party assistance.

| Property | Detail |
|---|---|
| Algorithm | K-Means clustering |
| Cloud dependency | ❌ Fully offline |
| Trigger | 2+ distinct speaker profiles detected |

---

## ⚙️ Getting Started

### Prerequisites

- Python **3.8+**
- A modern browser with microphone access
- ~1–2 GB disk space for Whisper model download (first run only)

### Setup & Run

**1. Open a terminal** in your project directory.

**2. Create a virtual environment:**
```bash
python -m venv venv
```

**3. Activate the environment:**
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

**5. Start the AI server:**
```bash
python app.py
```

**6. Open the dashboard:**

Navigate to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your browser.

> ⚠️ **First launch:** Whisper will download the model automatically. This is a one-time operation — subsequent starts are instant.

> 🔌 **Network note:** Run `python test_pyannote.py` to verify connectivity before starting if you encounter any model-loading issues.

---

## 🔑 Design Principles

- **Offline-first** — Speaker detection and emotion analysis run with zero cloud dependency
- **Parallel module execution** — All three AI modules process the audio stream independently and simultaneously
- **No per-request API cost** — Whisper runs locally after the initial model download
- **Single-command startup** — One `python app.py` brings the entire stack online

---

<div align="center">

**AI Interview & Assessment Monitoring System**

</div>
