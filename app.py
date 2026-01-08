from flask import Flask, render_template, Response, jsonify, request
import cv2
from collections import deque
import time
import random

from core.detector import HandDetector
from core.gestures import classify

app = Flask(__name__)

CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = HandDetector(max_hands=2)

HISTORY = deque(maxlen=7)

# ---------------- SECRET ACTIONS (stable edge) ----------------
ACTIONS = deque(maxlen=6)

# stable gate for action input
RAW_STABLE = {"g": "NONE", "count": 0}
STABLE_FRAMES = 4         # how many consecutive frames to consider "stable"
ACTION_GAP = 0.45         # min time gap between actions

LAST_ACTION_TIME = 0.0
LAST_UNLOCK_TIME = 0.0
UNLOCK_COOLDOWN = 2.0

# lock to keep UNDERSTOOD visible for a moment
SPECIAL_LOCK = {"g": "NONE", "until": 0.0}
SPECIAL_HOLD_SEC = 1.2

LAST_TWOHAND_TIME = 0.0
TWOHAND_WINDOW = 1.5

LAST = {"gesture": "NONE", "confidence": 0.0}

# ---------------- TRAINING MODE ----------------
TRAINING = {
    "enabled": False,
    "target": "STOP",
    "score": 0,
    "started_at": 0.0,
    "last_hit_at": 0.0,
}

TARGET_TO_GESTURE = {
    "STOP": "OPEN_PALM",
    "WAIT": "FIST",
    "CALM": "PEACE",
    "HELP": "HELP",
    "HANDS UP": "HANDS_UP",
    "UNDERSTOOD": "UNDERSTOOD",
    "TEAM READY": "TEAM_READY",
}

TRAINING_POOL = ["STOP", "WAIT", "CALM", "HANDS UP", "UNDERSTOOD", "TEAM READY", "HELP"]

def pick_target():
    return random.choice(TRAINING_POOL)

# ---------------- COMMAND LOG ----------------
EVENT_LOG = deque(maxlen=12)
LAST_LOG_GESTURE = "NONE"
LAST_LOG_TIME = 0.0
LOG_COOLDOWN = 0.7

def log_event(now, gesture, conf):
    global LAST_LOG_GESTURE, LAST_LOG_TIME
    if gesture == "NONE":
        return
    if gesture != LAST_LOG_GESTURE and (now - LAST_LOG_TIME) > LOG_COOLDOWN:
        ts = time.strftime("%H:%M:%S", time.localtime(now))
        EVENT_LOG.appendleft({"t": ts, "g": gesture, "c": round(float(conf), 2)})
        LAST_LOG_GESTURE = gesture
        LAST_LOG_TIME = now

def gen_frames():
    global LAST_TWOHAND_TIME, LAST_UNLOCK_TIME, LAST_ACTION_TIME

    while True:
        if not cap.isOpened():
            cap.open(CAM_INDEX)

        success, frame = cap.read()
        if not success:
            continue

        frame, hands = detector.process(frame)
        now = time.time()

        if hands and len(hands) > 0:
            cv2.putText(frame, f"Hands: {len(hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # classify first hand
            lms1 = hands[0]["lms"]
            label1 = hands[0]["label"]
            g1, conf1 = classify(lms1, label1)

            raw_frame_gesture = g1
            conf = conf1
            lms_for_help = lms1

            # two-hand HANDS_UP
            if len(hands) >= 2:
                lms2 = hands[1]["lms"]
                label2 = hands[1]["label"]
                g2, conf2 = classify(lms2, label2)

                cv2.putText(frame, f"H1:{g1}  H2:{g2}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if g1 == "OPEN_PALM" and g2 == "OPEN_PALM":
                    raw_frame_gesture = "HANDS_UP"
                    conf = (conf1 + conf2) / 2.0
                    LAST_TWOHAND_TIME = now

            # --------- smoothing (UI stability) ----------
            if raw_frame_gesture in ("OPEN_PALM", "FIST", "PEACE", "HANDS_UP"):
                HISTORY.append(raw_frame_gesture)
            else:
                HISTORY.append("NONE")

            smooth_gesture = max(set(HISTORY), key=HISTORY.count)
            final_gesture = smooth_gesture

            # --------- TEAM READY (2-hand -> peace) ----------
            if smooth_gesture == "PEACE" and (now - LAST_TWOHAND_TIME) < TWOHAND_WINDOW:
                final_gesture = "TEAM_READY"

            # --------- HELP (only for OPEN_PALM, avoid overriding specials) ----------
            if final_gesture not in ("UNDERSTOOD", "HANDS_UP", "TEAM_READY"):
                hand_top_y = min(pt[1] for pt in lms_for_help)
                cv2.putText(frame, f"top_y: {hand_top_y:.2f}", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if smooth_gesture == "OPEN_PALM" and hand_top_y < 0.30:
                    final_gesture = "HELP"

            # --------- UNDERSTOOD (more reliable) ----------
            # Use RAW stable detection for actions, not smoothed.
            if raw_frame_gesture in ("PEACE", "FIST", "OPEN_PALM"):
                if RAW_STABLE["g"] == raw_frame_gesture:
                    RAW_STABLE["count"] += 1
                else:
                    RAW_STABLE["g"] = raw_frame_gesture
                    RAW_STABLE["count"] = 1

                # when stable enough -> register action (edge)
                if RAW_STABLE["count"] == STABLE_FRAMES:
                    if (now - LAST_ACTION_TIME) > ACTION_GAP:
                        ACTIONS.append(raw_frame_gesture)
                        LAST_ACTION_TIME = now

                        # Trigger UNDERSTOOD on last 3 actions
                        if list(ACTIONS)[-3:] == ["PEACE", "FIST", "OPEN_PALM"]:
                            if (now - LAST_UNLOCK_TIME) > UNLOCK_COOLDOWN:
                                SPECIAL_LOCK["g"] = "UNDERSTOOD"
                                SPECIAL_LOCK["until"] = now + SPECIAL_HOLD_SEC
                                LAST_UNLOCK_TIME = now
                                ACTIONS.clear()
                                RAW_STABLE["count"] = 0
            else:
                # reset stability when other gestures appear
                RAW_STABLE["g"] = "NONE"
                RAW_STABLE["count"] = 0

            # Apply special lock (keeps UNDERSTOOD visible)
            if SPECIAL_LOCK["until"] > now:
                final_gesture = SPECIAL_LOCK["g"]
            else:
                if SPECIAL_LOCK["g"] != "NONE":
                    SPECIAL_LOCK["g"] = "NONE"

            # Debug actions
            cv2.putText(frame, f"actions: {list(ACTIONS)[-3:]}", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            LAST["gesture"] = final_gesture
            LAST["confidence"] = float(conf)

            # Command log
            log_event(now, final_gesture, conf)

            # -------- TRAINING CHECK --------
            if TRAINING["enabled"]:
                expected = TARGET_TO_GESTURE.get(TRAINING["target"], None)
                if expected and final_gesture == expected:
                    if now - TRAINING["last_hit_at"] > 1.0:
                        TRAINING["score"] += 1
                        TRAINING["last_hit_at"] = now
                        TRAINING["target"] = pick_target()

        else:
            HISTORY.clear()
            ACTIONS.clear()
            RAW_STABLE["g"] = "NONE"
            RAW_STABLE["count"] = 0
            SPECIAL_LOCK["g"] = "NONE"
            SPECIAL_LOCK["until"] = 0.0

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
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
        "HANDS_UP": {"label": "HANDS UP", "mood": "alert"},
        "TEAM_READY": {"label": "TEAM READY", "mood": "good"},
    }

    data = ui.get(g, {"label": g, "mood": "neutral"})

    return jsonify({
        "gesture": g,
        "confidence": round(conf, 2),
        "label": data["label"],
        "mood": data["mood"],
        "history": list(HISTORY),
        "actions": list(ACTIONS),
        "events": list(EVENT_LOG),
        "training": {
            "enabled": TRAINING["enabled"],
            "target": TRAINING["target"],
            "score": TRAINING["score"],
            "started_at": TRAINING["started_at"],
        }
    })

@app.route("/training/start", methods=["POST"])
def training_start():
    TRAINING["enabled"] = True
    TRAINING["score"] = 0
    TRAINING["started_at"] = time.time()
    TRAINING["last_hit_at"] = 0.0
    TRAINING["target"] = pick_target()
    return jsonify({"ok": True, "training": TRAINING})

@app.route("/training/stop", methods=["POST"])
def training_stop():
    TRAINING["enabled"] = False
    return jsonify({"ok": True, "training": TRAINING})

@app.route("/training/next", methods=["POST"])
def training_next():
    if TRAINING["enabled"]:
        TRAINING["target"] = pick_target()
    return jsonify({"ok": True, "training": TRAINING})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
