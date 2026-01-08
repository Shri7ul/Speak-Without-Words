from flask import Flask, render_template, Response, jsonify
import cv2

from core.detector import HandDetector
from core.gestures import classify

app = Flask(__name__)

CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = HandDetector(max_hands=1)

# shared state (simple for hackathon)
LAST = {"gesture": "NONE", "confidence": 0.0}

def gen_frames():
    while True:
        if not cap.isOpened():
            cap.open(CAM_INDEX)

        success, frame = cap.read()
        if not success:
            continue

        frame, all_hands = detector.process(frame)

        # update LAST
        if all_hands and len(all_hands) > 0:
            g, conf = classify(all_hands[0])
            LAST["gesture"] = g
            LAST["confidence"] = float(conf)
        else:
            LAST["gesture"] = "NONE"
            LAST["confidence"] = 0.0

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/predict")
def predict():
    # simple "visual language" mapping
    g = LAST["gesture"]
    conf = LAST["confidence"]

    ui = {
        "NONE":      {"label": "…", "mood": "neutral"},
        "OPEN_PALM": {"label": "STOP", "mood": "alert"},
        "FIST":      {"label": "HOLD", "mood": "focus"},
        "PEACE":     {"label": "PEACE", "mood": "calm"},
        "THUMB_UP":  {"label": "YES", "mood": "good"},
        "THUMB_DOWN":{"label": "NO", "mood": "bad"},
    }.get(g, {"label": g, "mood": "neutral"})

    return jsonify({
        "gesture": g,
        "confidence": round(conf, 2),
        "label": ui["label"],
        "mood": ui["mood"],
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
