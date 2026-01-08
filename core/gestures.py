# gesture mapping logic
import math

# MediaPipe landmark indices
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def fingers_up(lms):
    """
    lms: list[(x,y,z)] normalized
    Returns dict: finger->bool
    Rule is simple + works for many cases.
    """
    # For index/middle/ring/pinky: tip y < pip y => up (assuming upright-ish hand)
    up = {}
    up["index"]  = lms[TIP["index"]][1]  < lms[PIP["index"]][1]
    up["middle"] = lms[TIP["middle"]][1] < lms[PIP["middle"]][1]
    up["ring"]   = lms[TIP["ring"]][1]   < lms[PIP["ring"]][1]
    up["pinky"]  = lms[TIP["pinky"]][1]  < lms[PIP["pinky"]][1]

    # Thumb: compare x distance (works for many angles)
    # thumb tip far from palm (index mcp) => up
    thumb_tip = lms[TIP["thumb"]]
    index_mcp = lms[MCP["index"]]
    thumb_ip  = lms[3]
    up["thumb"] = _dist(thumb_tip, index_mcp) > _dist(thumb_ip, index_mcp)

    return up

def classify(lms):
    """
    Returns (gesture_name, confidence_float)
    Gestures: OPEN_PALM, FIST, PEACE, THUMB_UP, THUMB_DOWN, NONE
    """
    up = fingers_up(lms)

    total_up = sum(1 for k, v in up.items() if v)

    # OPEN PALM: all up
    if total_up >= 4 and up["index"] and up["middle"] and up["ring"] and up["pinky"]:
        return "OPEN_PALM", 0.85

    # FIST: none (or only thumb sometimes)
    if total_up <= 1 and (not up["index"]) and (not up["middle"]) and (not up["ring"]) and (not up["pinky"]):
        return "FIST", 0.80

    # PEACE: index + middle up, ring+pinky down
    if up["index"] and up["middle"] and (not up["ring"]) and (not up["pinky"]):
        return "PEACE", 0.82

    # THUMB UP / DOWN: only thumb up-ish but direction depends on thumb tip y vs wrist y
    if up["thumb"] and (not up["index"]) and (not up["middle"]) and (not up["ring"]) and (not up["pinky"]):
        wrist_y = lms[0][1]
        thumb_y = lms[TIP["thumb"]][1]
        if thumb_y < wrist_y:
            return "THUMB_UP", 0.80
        else:
            return "THUMB_DOWN", 0.78

    return "NONE", 0.40
