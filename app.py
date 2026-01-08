from flask import Flask, render_template, Response, jsonify
import cv2
from collections import deque
import time

from core.detector import HandDetector
from core.gestures import classify

app = Flask(__name__)

CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = HandDetector(max_hands=1)

# Smoothing history (frame-level)
HISTORY = deque(maxlen=7)

# Secret handshake (edge-based actions)
ACTIONS = deque(maxlen=5)
LAST_ACTION = "NONE"
LAST_ACTION_TIME = 0.0

LAST_UNLOCK_TIME = 0.0
UNLOCK_COOLDOWN = 2.0  # seconds

# Shared state
LAST = {"gesture": "NONE", "confidence": 0.0}

def gen_frames():
    global LAST_ACTION, LAST_ACTION_TIME, LAST_UNLOCK_TIME

    while True:
        if not cap.isOpened():
            cap.open(CAM_INDEX)

        success, frame = cap.read()
        if not success:
            continue

        frame, hands = detector.process(frame)

        if hands and len(hands) > 0:
            lms = hands[0]["lms"]
            hand_label = hands[0]["label"]

            # Debug label
            cv2.putText(frame, f"Hand: {hand_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            g, conf = classify(lms, hand_label)

            # Keep only reliable gestures
            if g in ("OPEN_PALM", "FIST", "PEACE"):
                HISTORY.append(g)
            else:
                HISTORY.append("NONE")

            # Majority vote smoothing
            final_gesture = max(set(HISTORY), key=HISTORY.count)

            # Hand top (for HELP detection)
            hand_top_y = min(pt[1] for pt in lms)  # 0=top, 1=bottom
            cv2.putText(frame, f"top_y: {hand_top_y:.2f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now = time.time()

            # ---- Secret Handshake: PEACE -> FIST -> OPEN_PALM ----
            # record only when gesture CHANGES (edge trigger) + debounce
            if final_gesture in ("PEACE", "FIST", "OPEN_PALM"):
                if final_gesture != LAST_ACTION and (now - LAST_ACTION_TIME) > 0.35:
                    ACTIONS.append(final_gesture)
                    LAST_ACTION = final_gesture
                    LAST_ACTION_TIME = now

                # check last 3 actions
                if list(ACTIONS)[-3:] == ["PEACE", "FIST", "OPEN_PALM"]:
                    if now - LAST_UNLOCK_TIME > UNLOCK_COOLDOWN:
                        final_gesture = "UNDERSTOOD"
                        LAST_UNLOCK_TIME = now
                        ACTIONS.clear()

            # ---- HELP (only if not UNDERSTOOD) ----
            if final_gesture != "UNDERSTOOD":
                if final_gesture == "OPEN_PALM" and hand_top_y < 0.30:
                    final_gesture = "HELP"

            # Optional debug: show actions on frame
            cv2.putText(frame, f"actions: {list(ACTIONS)[-3:]}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            LAST["gesture"] = final_gesture
            LAST["confidence"] = float(conf)

        else:
            HISTORY.clear()
            ACTIONS.clear()
            LAST_ACTION = "NONE"
            LAST_ACTION_TIME = 0.0

            LAST["gesture"] = "NONE"
            LAST["confidence"] = 0.0

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/predict")
def predict():
    g = LAST["gesture"]
    conf = LAST["confidence"]

    ui = {
        "NONE": {"label": "…", "mood": "neutral"},
        "OPEN_PALM": {"label": "STOP", "mood": "alert"},
        "FIST": {"label": "WAIT", "mood": "focus"},
        "PEACE": {"label": "CALM", "mood": "calm"},
        "HELP": {"label": "HELP", "mood": "alert"},
        "UNDERSTOOD": {"label": "UNDERSTOOD", "mood": "good"},
    }

    data = ui.get(g, {"label": g, "mood": "neutral"})

    return jsonify({
        "gesture": g,
        "confidence": round(conf, 2),
        "label": data["label"],
        "mood": data["mood"],
        "history": list(HISTORY),
        "actions": list(ACTIONS),   # ✅ extra debug/info
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
