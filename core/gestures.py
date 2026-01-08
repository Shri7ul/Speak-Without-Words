import math

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

def fingers_up(lms):
    up = {}
    up["index"]  = lms[TIP["index"]][1]  < lms[PIP["index"]][1]
    up["middle"] = lms[TIP["middle"]][1] < lms[PIP["middle"]][1]
    up["ring"]   = lms[TIP["ring"]][1]   < lms[PIP["ring"]][1]
    up["pinky"]  = lms[TIP["pinky"]][1]  < lms[PIP["pinky"]][1]
    return up

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

def fingers_up(lms):
    up = {}
    up["index"]  = lms[TIP["index"]][1]  < lms[PIP["index"]][1]
    up["middle"] = lms[TIP["middle"]][1] < lms[PIP["middle"]][1]
    up["ring"]   = lms[TIP["ring"]][1]   < lms[PIP["ring"]][1]
    up["pinky"]  = lms[TIP["pinky"]][1]  < lms[PIP["pinky"]][1]
    return up

def classify(lms, hand_label="Unknown"):
    up = fingers_up(lms)

    # OPEN PALM
    if up["index"] and up["middle"] and up["ring"] and up["pinky"]:
        return "OPEN_PALM", 0.85

    # FIST
    if (not up["index"]) and (not up["middle"]) and (not up["ring"]) and (not up["pinky"]):
        return "FIST", 0.80

    # PEACE
    if up["index"] and up["middle"] and (not up["ring"]) and (not up["pinky"]):
        return "PEACE", 0.82

    # ✅ THUMB UP/DOWN (more forgiving)
    # Require: other fingers mostly down (allow slight noise)
    other_down = (not up["index"]) and (not up["middle"]) and (not up["ring"]) and (not up["pinky"])

    thumb_tip = lms[TIP["thumb"]]
    wrist = lms[0]
    index_mcp = lms[5]
    pinky_mcp = lms[17]

    # thumb should be outside the palm horizontally relative to palm width
    palm_width = abs(index_mcp[0] - pinky_mcp[0]) + 1e-6
    thumb_far = (abs(thumb_tip[0] - index_mcp[0]) / palm_width) > 0.35 or (abs(thumb_tip[0] - pinky_mcp[0]) / palm_width) > 0.35

    if other_down and thumb_far:
        # up/down from y compared to wrist
        if thumb_tip[1] < wrist[1] - 0.03:
            return "THUMB_UP", 0.78
        if thumb_tip[1] > wrist[1] + 0.03:
            return "THUMB_DOWN", 0.76

    return "NONE", 0.40
