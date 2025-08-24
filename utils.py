import os, json, cv2
from moviepy.editor import ImageSequenceClip

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_alert(out_dir, frame_idx, timestamp_ms, score, boxes, labels):
    ensure_dir(out_dir)
    alert = {
        "frame_idx": int(frame_idx),
        "timestamp_ms": int(timestamp_ms),
        "score": float(score),
        "boxes": [list(map(float,b)) for b in boxes],
        "labels": labels
    }
    alerts_path = os.path.join(out_dir, "alerts.json")
    data = []
    if os.path.exists(alerts_path):
        try: data = json.load(open(alerts_path, "r"))
        except Exception: data = []
    data.append(alert)
    json.dump(data, open(alerts_path, "w"), indent=2)
    return alert

def draw_boxes(img, boxes, labels, color=(0,255,0)):
    for (x1,y1,x2,y2), lab in zip(boxes, labels):
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(img, lab, (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def write_snippet(out_dir, frames, fps, name):
    path = os.path.join(out_dir, "snippets"); ensure_dir(path)
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    fname = os.path.join(path, f"{name}.mp4")
    clip.write_videofile(fname, logger=None, audio=False)
    return fname
