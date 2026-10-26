import json
import sys
from analyze import analyze_video
from rules import SQUAT

LIMITS = {r["id"]: r["limit"] for r in SQUAT["rules"]}
BORDER = 8   # within 8 degrees of a limit = borderline


def run_agent(path, max_steps=3, video_out="output.mp4", snapshot=None):
    trace = []

    def log(step, call, obs, decision, reason, action):
        trace.append({"step": step, "tool_call": call, "observation": obs,
                      "decision": decision, "reason": reason, "action": action})
        print(f"\n--- Step {step} ---")
        print("Tool call  :", call)
        print("Observed   :", obs)
        print("Decision   :", decision, "-", reason)
        print("Action     :", action)

    min_conf, final, feedback = 0.3, None, []
    for step in range(1, max_steps + 1):
        r = analyze_video(path, min_conf=min_conf, write_video=video_out, snapshot=snapshot)
        obs = {"frames": r["frames"], "lost_frac": r["lost_frac"],
               "edge_frac": r["edge_frac"], "reps": len(r["reps"])}
        call = {"tool": "analyze_video", "min_conf": min_conf}

        if r["edge_frac"] > 0.25:
            final = "REQUEST_RETAKE"
            log(step, call, obs, final, "body touches the edge of the frame in many frames",
                "Ask user: step back so head-to-feet fits, then re-record.")
            break

        if r["lost_frac"] > 0.35:
            if min_conf > 0.15:
                log(step, call, obs, "RETRY_RELAXED", "many frames had no reliable pose",
                    "Re-run analysis with min_conf lowered to 0.15.")
                min_conf = 0.15
                continue
            final = "REQUEST_RETAKE"
            log(step, call, obs, final, "pose still unreliable after retry",
                "Ask user: improve lighting, use a plain background, keep full side view.")
            break

        if len(r["reps"]) == 0:
            final = "REQUEST_RETAKE"
            log(step, call, obs, final, "no complete squat rep detected",
                "Ask user: record from standing, squat down and back up, 3 reps.")
            break

        borderline = [x["rule"] for res in r["results"] for x in res
                      if abs(x["value"] - LIMITS[x["rule"]]) <= BORDER]
        feedback = [f"Rep {i}: {x['message']}" for i, res in enumerate(r["results"], 1)
                    for x in res if not x["ok"]] or ["All reps within limits."]
        relaxed = " (measured with relaxed confidence)" if min_conf < 0.3 else ""

        if borderline and len(r["reps"]) < 3:
            final = "REQUEST_MORE_REPS"
            log(step, call, obs, final,
                f"borderline on {sorted(set(borderline))}, only {len(r['reps'])} rep(s)",
                "Ask user: record 3 or more reps from the same side view.")
            break
        if borderline:
            final = "FLAG_HUMAN_REVIEW"
            log(step, call, obs, final, f"borderline on {sorted(set(borderline))}",
                "Give feedback but recommend a trainer check." + relaxed)
            break

        final = "FINALIZE"
        log(step, call, obs, final, "reliable pose and clear measurements",
            "Give coaching feedback." + relaxed)
        break

    with open("trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    print("Feedback   :", feedback)
    print("\nFinal outcome:", final)
    return {"outcome": final,
            "message": trace[-1]["action"] if trace else "",
            "feedback": feedback, "trace": trace}


if __name__ == "__main__":
    run_agent(sys.argv[1] if len(sys.argv) > 1 else "clips/squat1.mp4")