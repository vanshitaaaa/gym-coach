import sys
import cv2
import numpy as np

MODEL = "models/yolov8n-pose.onnx"
SIZE = 640
net = cv2.dnn.readNetFromONNX(MODEL)

EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16)]

def letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top

def detect_pose(frame, conf_thres=0.5):
    img, r, left, top = letterbox(frame)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (SIZE, SIZE), swapRB=True)
    net.setInput(blob)
    out = net.forward()[0].T          # (8400, 56)
    best = int(np.argmax(out[:, 4]))
    if out[best, 4] < conf_thres:
        return None
    kps = out[best, 5:].reshape(17, 3).copy()
    kps[:, 0] = (kps[:, 0] - left) / r
    kps[:, 1] = (kps[:, 1] - top) / r
    return kps, float(out[best, 4])

def draw(frame, kps):
    for a, b in EDGES:
        if kps[a, 2] > 0.3 and kps[b, 2] > 0.3:
            cv2.line(frame, tuple(kps[a, :2].astype(int)),
                     tuple(kps[b, :2].astype(int)), (0, 255, 0), 3)
    for x, y, c in kps:
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    return frame

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "images/test.jpg"
    img = cv2.imread(path)
    if img is None:
        sys.exit("Could not read " + path)
    result = detect_pose(img)
    if result is None:
        sys.exit("No person found")
    kps, score = result
    print("Person confidence:", round(score, 2))
    names = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
             "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
             "l_knee","r_knee","l_ankle","r_ankle"]
    for n, (x, y, c) in zip(names, kps):
        print(f"{n:11s} x={x:7.1f} y={y:7.1f} conf={c:.2f}")
    cv2.imwrite("output.jpg", draw(img, kps))
    print("Saved output.jpg")