# 🖐️ Speak Without Words  
### Real-time Gesture → Intent → Visual Language

> A real-time computer vision system that allows humans to communicate **without speaking**, using hand gestures captured via webcam and translated into visual + voice feedback.

---

## 🚀 Inspiration
Many people cannot communicate verbally in critical situations — due to noise, disability, distance, or urgency.  
**Speak Without Words** explores how **hand gestures alone** can become a universal, fast, and intuitive communication layer.

---

## ✨ Features

### 🧠 Core
- Real-time hand detection using **OpenCV + MediaPipe**
- Gesture classification (rule-based, no heavy ML)
- Gesture smoothing (majority voting)
- Confidence scoring

### ✋ Supported Gestures
| Gesture | Meaning |
|------|------|
| Open Palm | ⛔ STOP |
| Fist | ✋ WAIT |
| Peace | 💙 CALM |
| Open Palm (High) | 🚨 HELP |
| Both Palms | 🙌 HANDS UP |

---

### 🔐 Secret Gesture Codes
- **UNDERSTOOD** → `PEACE → FIST → OPEN PALM`
- **TEAM READY** → `HANDS UP → PEACE`

> These are detected using **temporal gesture sequences**, not single frames.

---

### 🔊 Voice Feedback
- Browser-native **Text-to-Speech**
- Speaks detected intent (STOP, HELP, TEAM READY, etc.)
- Fully client-side (no audio files required)

*(Optional sound effect support included, commented in code)*

---

### 🏋️ Training Mode
- Practice gestures interactively
- Random target gestures
- Live scoring system
- Helps users learn gesture language quickly

---

### 📜 Command Log
- Shows last detected commands
- Includes timestamp + confidence
- Only logs **gesture changes** (no spam)

---

### 🎨 UI / UX
- Clean **Tailwind CSS** UI
- Live confidence bar
- Animated banners for critical actions
- Dev mode for debugging
- Works fully in browser (no frontend framework)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Vision | OpenCV, MediaPipe |
| Backend | Python, Flask |
| Frontend | HTML, Tailwind CSS, Vanilla JS |
| Voice | Web Speech API |
| Architecture | Real-time streaming (MJPEG) |

---

## 📂 Project Structure

```yaml
Speak-Without-Words/
│
├── app.py                  # Flask backend
├── core/
│   ├── detector.py         # Hand detection (MediaPipe)
│   └── gestures.py         # Gesture classification logic
│
├── templates/
│   └── index.html          # UI (Tailwind + JS)
│
├── static/
│   └── sounds.mp3          # (Optional) sound effect
│
├── requirements.txt
└── README.md
````

---

# Speak-Without-Words

## ⚙️ Installation & Run

### 1️⃣ Create environment
```bash
conda create -n speak-without-words python=3.11 -y
```
```bash
conda activate speak-without-words
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app
```bash
python app.py
```

### 3️⃣ Open in browser

```bash
http://127.0.0.1:5000
```

---

## 🎯 Use Cases

* Silent communication in noisy environments
* Accessibility tools for speech-impaired users
* Emergency signaling
* Human–computer interaction research
* Gesture-based UI systems

---

## 🧪 How It Works (High Level)

1. Webcam captures frames
2. MediaPipe detects hand landmarks
3. Rule-based logic classifies gestures
4. Temporal smoothing stabilizes output
5. Gesture → intent → UI + voice feedback

---

## 🏆 Hackathon Notes

* Built rapidly with focus on **clarity + usability**
* No heavy ML training required
* Runs fully on local machine
* Easy to demo and explain

---

## 🔮 Future Improvements

* Multi-user gesture detection
* Sign-language sentence building
* Mobile camera support
* Gesture personalization
* Cloud / IoT integrations

---

## 🧑‍💻 Author
**Shriful Islam** (InHuman)  

Built with ❤️ by a university student exploring **Computer Vision, AI, and Human-Centered Design**.

---

## 📜 License

MIT License — free to use, modify, and extend.

