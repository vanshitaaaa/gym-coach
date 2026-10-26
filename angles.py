import numpy as np

LEFT = dict(shoulder=5, hip=11, knee=13, ankle=15)
RIGHT = dict(shoulder=6, hip=12, knee=14, ankle=16)

def angle(a, b, c):
    """Angle in degrees at point b, formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def trunk_lean(shoulder, hip):
    """Torso angle from vertical in degrees (0 = upright)."""
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]      # image y points downward
    return float(np.degrees(np.arctan2(abs(dx), dy)))

def pick_side(kps):
    left = sum(kps[i, 2] for i in LEFT.values())
    right = sum(kps[i, 2] for i in RIGHT.values())
    return ("left", LEFT) if left >= right else ("right", RIGHT)

def frame_metrics(kps, min_conf=0.3):
    side, idx = pick_side(kps)
    if min(kps[i, 2] for i in idx.values()) < min_conf:
        return None                 # not confident enough
    p = {k: kps[i, :2] for k, i in idx.items()}
    return {
        "side": side,
        "knee": angle(p["hip"], p["knee"], p["ankle"]),
        "hip": angle(p["shoulder"], p["hip"], p["knee"]),
        "lean": trunk_lean(p["shoulder"], p["hip"]),
    }