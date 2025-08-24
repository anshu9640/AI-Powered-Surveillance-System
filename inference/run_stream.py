# inference/run_stream.py
import os, glob, cv2, argparse, json
import numpy as np
import torch
from ultralytics import YOLO

from models.autoencoder3d import AE3D, reconstruction_error
from fusion import SimpleTracker, loitering_flags, AbandonmentWatcher
from utils import ensure_dir, draw_boxes, save_alert, write_snippet

IMG_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

# -------- helpers --------
def preprocess_gray112(frames):
    gs = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    gs = [cv2.resize(g, (112, 112), interpolation=cv2.INTER_AREA) for g in gs]
    arr = np.stack(gs, axis=0).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,T,H,W)
    arr = np.expand_dims(arr, axis=0)  # (1,1,T,H,W)
    return torch.from_numpy(arr)

def frames_from_source(source, default_fps):
    """Yield (frame, fps) from a folder of images or a video file."""
    if os.path.isdir(source):
        files = sorted(
            [p for p in glob.glob(os.path.join(source, "*"))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
        )
        if not files:
            raise FileNotFoundError(f"No image files found in folder: {source}")
        fps = default_fps
        for p in files:
            img = cv2.imread(p)
            if img is not None:
                yield img, fps
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
        ok, f = cap.read()
        while ok:
            yield f, fps
            ok, f = cap.read()
        cap.release()

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a video FILE or an image-sequence FOLDER")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ae", default="models/ae3d.pth")
    ap.add_argument("--fps", type=int, default=25, help="fps when reading from an image folder")

    # Detection controls
    ap.add_argument("--det_conf", type=float, default=0.40, help="YOLO confidence threshold")
    ap.add_argument("--semantic_labels", type=str,
                    default="bicycle,skateboard,car,truck,bus",
                    help="Labels that count as semantic anomalies (comma-separated)")
    ap.add_argument("--ignore_labels", type=str, default="",
                    help="Labels to ignore entirely (comma-separated)")

    # Abandonment controls
    ap.add_argument("--abandon_labels", type=str, default="backpack,handbag,suitcase",
                    help="Labels considered for abandonment (comma-separated)")
    ap.add_argument("--abandon_seconds", type=float, default=5.0)
    ap.add_argument("--abandon_radius", type=float, default=10.0, help="px movement radius considered 'stationary'")
    ap.add_argument("--abandon_assoc_iou", type=float, default=0.2, help="IoU to associate bag with a person")

    # Fusion weights
    ap.add_argument("--semantic_weight", type=float, default=0.45)
    ap.add_argument("--motion_weight", type=float, default=0.35)
    ap.add_argument("--loiter_weight", type=float, default=0.10)
    ap.add_argument("--abandon_weight", type=float, default=0.25)

    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    allow_sem = {s.strip().lower() for s in args.semantic_labels.split(",") if s.strip()}
    ignore = {s.strip().lower() for s in args.ignore_labels.split(",") if s.strip()}
    allow_abandon = {s.strip().lower() for s in args.abandon_labels.split(",") if s.strip()}

    ensure_dir(args.out); ensure_dir(os.path.join(args.out, "frames")); ensure_dir(os.path.join(args.out, "snippets"))

    # Models
    det = YOLO("yolov8n.pt")
    ae = AE3D().to(args.device)
    if os.path.exists(args.ae):
        ae.load_state_dict(torch.load(args.ae, map_location=args.device))
    ae.eval()

    frame_buffer, snippet_buffer = [], []
    tracker = SimpleTracker()
    abandon = AbandonmentWatcher(min_seconds=args.abandon_seconds,
                                 fps=args.fps,
                                 stationary_radius=args.abandon_radius,
                                 assoc_iou=args.abandon_assoc_iou,
                                 forget_after=int(args.fps * 3))

    t, cooldown = 0, 0

    def run_yolo(frame):
        r = det.predict(frame, imgsz=640, conf=args.det_conf, verbose=False)[0]
        boxes, names, confs = [], [], []
        for b, c, conf in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.cls.cpu().numpy(),
                              r.boxes.conf.cpu().numpy()):
            name = det.names[int(c)].lower()
            if name in ignore:
                continue
            boxes.append(b.tolist()); names.append(name); confs.append(float(conf))
        return boxes, names, confs

    for f, fps in frames_from_source(args.video, args.fps):
        t += 1
        # sync abandonment window to true fps (for videos)
        if t == 1:
            abandon.fps = fps
            abandon.win = int(args.abandon_seconds * fps)

        ts_ms = int(1000.0 * (t / fps))
        annotated = f.copy()

        # 1) YOLO detections
        boxes, names, confs = run_yolo(f)
        labels_draw = [f"{n}:{c:.2f}" for n, c in zip(names, confs)]
        draw_boxes(annotated, boxes, labels_draw, (0, 255, 0))

        # People / bag separation
        person_boxes = [b for b, n in zip(boxes, names) if n == "person"]
        bag_boxes = [b for b, n in zip(boxes, names) if n in allow_abandon]

        # Semantic anomaly if any allowed class is present
        semantic_score = 1.0 if any(n in allow_sem for n in names) else 0.0

        # 2) Loitering via centroid tracker
        dets_for_track = [(x1, y1, x2, y2, n, 1.0) for (x1, y1, x2, y2), n in zip(boxes, names)]
        tracks = tracker.update(dets_for_track, t)
        loiter_ids = loitering_flags(tracks, min_seconds=3.0, fps=fps, min_radius=12.0)
        rule_loiter = len(loiter_ids) > 0
        if rule_loiter:
            cv2.putText(annotated, "LOITER", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 3) Abandoned object rule
        abandoned_boxes = abandon.update(bag_boxes, person_boxes, t)
        rule_abandon = len(abandoned_boxes) > 0
        for bx in abandoned_boxes:
            x1, y1, x2, y2 = map(int, bx)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, "ABANDONED", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4) AE motion anomaly (every 16 frames)
        frame_buffer.append(f); snippet_buffer.append(annotated.copy())
        motion_score = 0.0
        if len(frame_buffer) >= 16:
            clip = preprocess_gray112(frame_buffer[-16:])
            with torch.no_grad():
                x = clip.to(args.device); xhat = ae(x)
                err = reconstruction_error(x, xhat).item()
                motion_score = float(np.clip(err * 20.0, 0.0, 1.0))  # heuristic scaling

        # 5) Fuse scores
        fuse = (args.semantic_weight * semantic_score +
                args.motion_weight   * motion_score   +
                args.loiter_weight   * (1.0 if rule_loiter  else 0.0) +
                args.abandon_weight  * (1.0 if rule_abandon else 0.0))
        score = float(np.clip(fuse, 0.0, 1.0))

        # overlay HUD
        cv2.putText(
            annotated,
            f"S:{semantic_score:.2f} M:{motion_score:.2f} "
            f"L:{1.0 if rule_loiter else 0.0:.2f} A:{1.0 if rule_abandon else 0.0:.2f} "
            f"F:{score:.2f}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # 6) Trigger alert
        if score > args.threshold and cooldown == 0:
            aid = f"alert_{t:06d}"
            cv2.imwrite(os.path.join(args.out, "frames", f"{aid}.jpg"), annotated)

            # persist alert + per-component breakdown
            extra = []
            if rule_loiter:  extra.append("loiter")
            if rule_abandon: extra.append("abandoned_object")

            _ = save_alert(args.out, t, ts_ms, score, boxes, names + extra)

            # attach exact components to the last alert saved
            alerts_path = os.path.join(args.out, "alerts.json")
            data = json.load(open(alerts_path, "r"))
            data[-1]["components"] = {
                "semantic": float(semantic_score),
                "motion":   float(motion_score),
                "loiter":   1.0 if rule_loiter   else 0.0,
                "abandon":  1.0 if rule_abandon  else 0.0,
            }
            json.dump(data, open(alerts_path, "w"), indent=2)

            # short snippet (~2s)
            n = min(len(snippet_buffer), int(fps * 2))
            write_snippet(args.out, snippet_buffer[-n:], int(fps), aid)

            cooldown = int(fps * 1.5)

        # periodic preview frames
        if cooldown > 0:
            cooldown -= 1
        if fps >= 2 and (t % max(int(fps // 2), 1) == 0):
            cv2.imwrite(os.path.join(args.out, "frames", f"preview_{t:06d}.jpg"), annotated)

    print(f"Done. Alerts at {os.path.join(args.out, 'alerts.json')}")
    if not os.path.exists(os.path.join(args.out, "alerts.json")):
        json.dump([], open(os.path.join(args.out, "alerts.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
