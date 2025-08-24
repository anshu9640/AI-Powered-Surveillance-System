# fusion.py
import numpy as np
from collections import deque

# ---------------- Centroid Tracker (for loitering) ----------------
class SimpleTracker:
    """Tiny centroid tracker for motion & loitering."""
    def __init__(self, max_lost=15, dist_thresh=80):
        self.next_id = 1
        self.tracks = {}  # id -> {'xy': (x,y), 'lost': 0, 'history': deque([(x,y,t),...])}
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def _dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def update(self, detections, t):
        # detections: list[(x1,y1,x2,y2,cls,conf)]
        centroids = [((x1+x2)/2.0, (y1+y2)/2.0) for x1,y1,x2,y2,cls,conf in detections]
        assigned = set()
        for tid, tr in list(self.tracks.items()):
            if not centroids:
                tr['lost'] += 1
                continue
            dists = [self._dist(tr['xy'], c) for c in centroids]
            j = int(np.argmin(dists))
            if dists[j] < self.dist_thresh and j not in assigned:
                c = centroids[j]
                tr['xy'] = c
                tr['lost'] = 0
                tr['history'].append((c[0], c[1], t))
                assigned.add(j)
            else:
                tr['lost'] += 1
        for j, c in enumerate(centroids):
            if j in assigned:
                continue
            self.tracks[self.next_id] = {'xy': c, 'lost': 0, 'history': deque(maxlen=60)}
            self.tracks[self.next_id]['history'].append((c[0], c[1], t))
            self.next_id += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]
        return self.tracks

def loitering_flags(tracks, min_seconds=3.0, fps=25, min_radius=12.0):
    """Flag IDs that barely move over a time window."""
    flags = set()
    win = int(min_seconds * fps)
    for tid, tr in tracks.items():
        hist = list(tr['history'])
        if len(hist) >= win:
            xs = [p[0] for p in hist[-win:]]
            ys = [p[1] for p in hist[-win:]]
            cx, cy = np.mean(xs), np.mean(ys)
            rad = np.mean([np.hypot(x-cx, y-cy) for x,y in zip(xs,ys)])
            if rad < min_radius:
                flags.add(tid)
    return flags

# ---------------- Abandoned Object Watcher ----------------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(ua, 1e-6)

class AbandonmentWatcher:
    """
    Tracks candidate objects (e.g., bag-like classes). 
    If an object stays nearly stationary for >= min_seconds AND has no overlapping person 
    for that duration, it's flagged as 'abandoned'.
    """
    def __init__(self, min_seconds=5.0, fps=25, stationary_radius=10.0, assoc_iou=0.2, forget_after=60):
        self.fps = fps
        self.win = int(min_seconds * fps)
        self.radius = stationary_radius
        self.assoc_iou = assoc_iou
        self.forget_after = forget_after
        self.next_id = 1
        # id -> state
        self.items = {}  # {'box':(x1,y1,x2,y2), 'hist':deque([(cx,cy,t)]), 'last':t, 'no_person_count':int, 'abandoned':bool}

    def _match(self, boxes):
        # greedy IOU matching
        matches = [-1]*len(boxes)
        used = set()
        for iid, st in self.items.items():
            best_j, best_iou = -1, 0.0
            for j,b in enumerate(boxes):
                if j in used: continue
                iou = _iou(st['box'], b)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= 0.3:
                matches[best_j] = iid
                used.add(best_j)
        return matches

    def update(self, obj_boxes, person_boxes, t):
        # 1) associate
        matches = self._match(obj_boxes)
        # 2) update / create
        for j,b in enumerate(obj_boxes):
            if matches[j] == -1:
                iid = self.next_id; self.next_id += 1
                self.items[iid] = {
                    'box': b, 'hist': deque(maxlen=self.win), 'last': t,
                    'no_person_count': 0, 'abandoned': False
                }
                cx,cy = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0
                self.items[iid]['hist'].append((cx,cy,t))
            else:
                iid = matches[j]
                st = self.items[iid]
                st['box'] = b; st['last'] = t
                cx,cy = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0
                st['hist'].append((cx,cy,t))

        # 3) decide abandonment per item
        abandoned_boxes = []
        for iid, st in list(self.items.items()):
            # proximity to any person?
            near_person = any(_iou(st['box'], pb) >= self.assoc_iou for pb in person_boxes)
            st['no_person_count'] = 0 if near_person else min(st['no_person_count']+1, self.win*2)

            # stationary?
            hist = list(st['hist'])
            stationary = False
            if len(hist) >= self.win:
                xs = [p[0] for p in hist[-self.win:]]
                ys = [p[1] for p in hist[-self.win:]]
                cx,cy = np.mean(xs), np.mean(ys)
                rad = np.mean([np.hypot(x-cx, y-cy) for x,y in zip(xs,ys)])
                stationary = rad < self.radius

            if stationary and st['no_person_count'] >= self.win:
                st['abandoned'] = True
                abandoned_boxes.append(st['box'])

            # forget stale tracks
            if t - st['last'] > self.forget_after:
                del self.items[iid]

        return abandoned_boxes
