# Each exercise is just a dictionary: thresholds + rules.
SQUAT = {
    "name": "squat",
    "camera_view": "side",
    "down_knee": 120,   # knee angle below this = going down
    "up_knee": 160,     # knee angle above this = back up, rep finished
    "rules": [
        {
            "id": "depth",
            "metric": "min_knee",      # deepest knee angle in the rep
            "ok_if": "<=",
            "limit": 100,
            "ok": "Good depth.",
            "bad": "Too shallow. Lower until your thighs are close to parallel with the floor.",
        },
        {
            "id": "lean",
            "metric": "max_lean",      # biggest forward torso lean in the rep
            "ok_if": "<=",
            "limit": 50,
            "ok": "Torso position is good.",
            "bad": "Leaning too far forward. Keep your chest up and torso more upright.",
        },
    ],
}

def evaluate_rep(rep, exercise):
    """Check one rep against every rule of the exercise."""
    results = []
    for r in exercise["rules"]:
        value = rep[r["metric"]]
        ok = value <= r["limit"] if r["ok_if"] == "<=" else value >= r["limit"]
        results.append({
            "rule": r["id"],
            "value": round(value),
            "ok": ok,
            "message": r["ok"] if ok else r["bad"],
        })
    return results