import sys
import cv2
from pose import detect_pose, draw
from angles import frame_metrics, pick_side
from rules import SQUAT, evaluate_rep


def analyze_video(path, exercise=SQUAT, min_conf=0.3, write_video=None, snapshot=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if write_video:
        out = cv2.VideoWriter(write_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    reps, state, cur = [], "up", None
    frames = lost = edge = 0
    banner = ""
    best_k, best_frame = 999, None   # deepest-squat frame for the snapshot

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        res = detect_pose(frame)
        m = frame_metrics(res[0], min_conf=min_conf) if res else None

        if res:
            # is the body touching the edge of the picture?
            _, idx = pick_side(res[0])
            pts = res[0][list(idx.values()), :2]
            if ((pts[:, 0] < 0.02 * w).any() or (pts[:, 0] > 0.98 * w).any()
                    or (pts[:, 1] < 0.02 * h).any() or (pts[:, 1] > 0.98 * h).any()):
                edge += 1

        if m is None:
            lost += 1
        else:
            draw(frame, res[0])
            k = m["knee"]
            if state == "up" and k < exercise["down_knee"]:
                state = "down"
                cur = {"min_knee": k, "max_lean": m["lean"]}
            elif state == "down":
                cur["min_knee"] = min(cur["min_knee"], k)
                cur["max_lean"] = max(cur["max_lean"], m["lean"])
                if k > exercise["up_knee"]:
                    reps.append(cur)
                    banner = f"Rep {len(reps)} done"
                    state = "up"
            cv2.putText(frame, f"knee {k:.0f}  lean {m['lean']:.0f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if banner:
            cv2.putText(frame, banner, (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        if out:
            out.write(frame)

        # remember the frame where the knee was most bent
        if m is not None and m["knee"] < best_k:
            best_k, best_frame = m["knee"], frame.copy()

    cap.release()
    if out:
        out.release()
    if snapshot and best_frame is not None:
        cv2.imwrite(snapshot, best_frame)

    return {
        "frames": frames,
        "lost_frac": round(lost / max(frames, 1), 2),
        "edge_frac": round(edge / max(frames, 1), 2),
        "reps": reps,
        "results": [evaluate_rep(r, exercise) for r in reps],
        "min_conf": min_conf,
    }


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "clips/squat1.mp4"
    r = analyze_video(p, write_video="output.mp4", snapshot="snapshot.jpg")
    print({k: v for k, v in r.items() if k not in ("reps", "results")})
    for i, res in enumerate(r["results"], 1):
        print(f"Rep {i}:", [(x["rule"], x["value"], x["ok"]) for x in res])