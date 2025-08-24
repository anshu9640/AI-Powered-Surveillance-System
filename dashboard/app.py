# dashboard/app.py
import os, json, glob, io
from datetime import timedelta
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI-Powered Surveillance — Review",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styles ----------
st.markdown(
    """
    <style>
      .metric-card{
        padding:14px 16px;border-radius:16px;background:linear-gradient(180deg,#0b1220,#0a1020);
        color:#ecfeff;border:1px solid #1f2937; box-shadow:0 2px 12px rgba(0,0,0,.25);
      }
      .metric-value{font-size:26px;font-weight:800;line-height:1;margin-top:-4px;}
      .metric-label{font-size:12px;opacity:.8}
      .pill{padding:2px 8px;border-radius:999px;background:#111827;color:#93c5fd;border:1px solid #1f2937;font-size:12px}
      .badge{padding:2px 8px;border-radius:999px;background:#0ea5e9;color:#fff;font-size:12px}
      .small{font-size:12px;color:#9ca3af}
      .card{padding:12px;border-radius:14px;background:#0b1220;border:1px solid #1f2937}
      .help{border-left:4px solid #60a5fa;padding-left:10px}
      .muted{color:#9ca3af}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📹 AI-Powered Surveillance — Alerts Review")

# ---------- Helpers ----------
ALLOWED_SEMANTIC = {"bicycle","skateboard","car","truck","bus","motorcycle","cart"}
def list_out_dirs():
    here = "."
    return sorted([d for d in os.listdir(here) if os.path.isdir(d) and os.path.exists(os.path.join(d,"alerts.json"))])

def load_alerts(out_dir):
    path = os.path.join(out_dir, "alerts.json")
    try:
        data = json.load(open(path))
    except Exception:
        data = []
    # normalize
    for a in data:
        a["frame_idx"] = int(a.get("frame_idx", 0))
        a["timestamp_ms"] = int(a.get("timestamp_ms", 0))
        a["score"] = float(a.get("score", 0.0))
        a["labels"] = a.get("labels", [])
    return data

def kpi(label, value):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)

def secs(ms): return ms/1000.0

def timeline_plot(alerts):
    xs = [a["frame_idx"] for a in alerts]
    ys = [a["score"] for a in alerts]
    fig, ax = plt.subplots(figsize=(10,2.6))
    ax.scatter(xs, ys, s=26, alpha=0.9)
    ax.plot(xs, ys, linewidth=1, alpha=0.55)
    ax.set_ylim(0,1.02)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fused score")
    ax.grid(True, linestyle="--", alpha=0.25)
    st.pyplot(fig, clear_figure=True)

def chips(labels):
    if not labels: 
        st.markdown("<span class='muted small'>no labels</span>", unsafe_allow_html=True)
        return
    st.markdown(" ".join([f"<span class='pill'>{stx}</span>" for stx in labels]), unsafe_allow_html=True)

def component_bars(S, M, L, A, wS, wM, wL, wA):
    # small explanatory bar chart for components
    fig, ax = plt.subplots(figsize=(4.6,2.4))
    names = ["Semantic","Motion","Loiter","Abandon"]
    vals  = [S, M, L, A]
    ax.bar(names, vals)
    ax.set_ylim(0,1.02)
    ax.set_ylabel("Component (0–1)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    st.pyplot(fig, clear_figure=True)
    fused = wS*S + wM*M + wL*L + wA*A
    st.markdown(f"**Fused score (estimate)** = {wS:.2f}·S + {wM:.2f}·M + {wL:.2f}·L + {wA:.2f}·A = **{min(max(fused,0.0),1.0):.2f}**")

def explain_alert(alert, wS, wM, wL, wA):
    """Heuristic explainer using saved labels + weights.
       If your alerts.json includes 'components', we use those directly;
       otherwise we estimate Motion so the story still makes sense."""
    labels = [str(l).lower() for l in alert.get("labels",[])]
    comps = alert.get("components", None)  # optional if you added it in inference
    # Boolean rules from labels
    has_loiter   = "loiter" in labels
    has_abandon  = "abandoned_object" in labels
    has_semantic = any(l in ALLOWED_SEMANTIC for l in labels)

    if comps:
        S = float(comps.get("semantic", 1.0 if has_semantic else 0.0))
        M = float(comps.get("motion",   0.0))
        L = float(comps.get("loiter",   1.0 if has_loiter else 0.0))
        A = float(comps.get("abandon",  1.0 if has_abandon else 0.0))
    else:
        # Estimate motion so that weights roughly match the fused score
        S = 1.0 if has_semantic else 0.0
        L = 1.0 if has_loiter else 0.0
        A = 1.0 if has_abandon else 0.0
        known = wS*S + wL*L + wA*A
        rem = max(0.0, alert["score"] - known)
        M = max(0.0, min(1.0, rem / max(wM, 1e-6)))  # clamp to [0,1]

    # Narrative
    bullets = []
    if has_semantic:
        bullets.append("**Semantic:** non-pedestrian object(s) detected (e.g., bicycle/vehicle).")
    else:
        bullets.append("**Semantic:** none of the configured non-pedestrian classes were seen.")
    if has_loiter:
        bullets.append("**Loitering:** a tracked person stayed within a small radius for multiple seconds.")
    if has_abandon:
        bullets.append("**Abandoned object:** a bag-like object remained without a nearby person for several seconds.")
    if not has_loiter and not has_abandon and not has_semantic:
        bullets.append("No rule fired; this alert likely came from **unusual motion**.")

    st.markdown("**Why this alert?**")
    st.markdown("<div class='card help'>"+ "<br/>".join(f"• {b}" for b in bullets) +"</div>", unsafe_allow_html=True)

    component_bars(S, M, L, A, wS, wM, wL, wA)


# ---------- Sidebar (controls) ----------
with st.sidebar:
    st.header("⚙️ Controls")
    preset = list_out_dirs()
    default_dir = preset[0] if preset else "out_peds1"
    choose = st.selectbox("Choose output folder", preset or [default_dir], index=0 if preset else 0)
    out_dir = st.text_input("…or type a custom folder", value=choose)

    st.markdown("<div class='small muted'>Filter</div>", unsafe_allow_html=True)
    min_score = st.slider("Minimum alert score", 0.0, 1.0, 0.50, 0.01)
    sort_by = st.radio("Sort alerts by", ["time ↑", "time ↓", "score ↑", "score ↓"], horizontal=True)

    st.markdown("---")
    st.markdown("<div class='small muted'>Scoring weights (for explanations)</div>", unsafe_allow_html=True)
    wS = st.slider("Semantic (S)", 0.0, 1.0, 0.45, 0.05)
    wM = st.slider("Motion (M)",   0.0, 1.0, 0.35, 0.05)
    wL = st.slider("Loiter (L)",   0.0, 1.0, 0.10, 0.05)
    wA = st.slider("Abandon (A)",  0.0, 1.0, 0.25, 0.05)
    # normalize (optional)
    total = max(wS+wM+wL+wA, 1e-6)
    wS, wM, wL, wA = wS/total, wM/total, wL/total, wA/total

    st.markdown(
        "<span class='small muted'>Tip: set these to match the weights you used in inference for best explanations.</span>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("Previews are saved at `frames/preview_*.jpg`, snapshots at `frames/alert_*.jpg`, and clips in `snippets/`.")

# ---------- Data load ----------
if not os.path.isdir(out_dir):
    st.warning("Enter a valid output folder that contains an `alerts.json`.")
    st.stop()

alerts = load_alerts(out_dir)
all_labels = sorted({lbl for a in alerts for lbl in a.get("labels", [])})
label_filter = st.multiselect("Filter by label(s)", all_labels, default=[])

# Apply filters
filtered = [a for a in alerts if a["score"] >= min_score]
if label_filter:
    filtered = [a for a in filtered if any(l in label_filter for l in a.get("labels", []))]

# sort
if sort_by == "time ↑":   filtered = sorted(filtered, key=lambda a: a["frame_idx"])
elif sort_by == "time ↓": filtered = sorted(filtered, key=lambda a: a["frame_idx"], reverse=True)
elif sort_by == "score ↑": filtered = sorted(filtered, key=lambda a: a["score"])
else:                      filtered = sorted(filtered, key=lambda a: a["score"], reverse=True)

# ---------- Tabs ----------
tab_overview, tab_alerts, tab_gallery, tab_about = st.tabs(["Overview", "Alerts", "Gallery", "About"])

# Overview
with tab_overview:
    left, mid, right, right2 = st.columns([1,1,1,1], gap="small")
    total = len(filtered)
    avg = (sum(a["score"] for a in filtered)/total) if total else 0.0
    mx = max([a["score"] for a in filtered]) if total else 0.0
    span = 0
    if total:
        t0 = min(a["timestamp_ms"] for a in filtered)
        t1 = max(a["timestamp_ms"] for a in filtered)
        span = secs(t1 - t0)

    with left:  kpi("Total Alerts (filtered)", f"{total}")
    with mid:   kpi("Avg Score", f"{avg:.2f}")
    with right: kpi("Max Score", f"{mx:.2f}")
    with right2:kpi("Time Span", f"{timedelta(seconds=int(span))}")

    st.subheader("Timeline")
    if filtered:
        timeline_plot(filtered)
    else:
        st.info("No alerts match your filters.")

# Alerts
with tab_alerts:
    if not filtered:
        st.info("No alerts to review — try lowering the score filter.")
    else:
        idx_map = {f"#{i+1} • frame {a['frame_idx']} • score {a['score']:.2f}": i for i,a in enumerate(filtered)}
        choice = st.selectbox("Browse alerts", list(idx_map.keys()), index=0)
        sel = filtered[idx_map[choice]]

        c1, c2 = st.columns([2,1], gap="large")
        aid = f"alert_{sel['frame_idx']:06d}"
        frame_img = os.path.join(out_dir, "frames", f"{aid}.jpg")
        clip_mp4  = os.path.join(out_dir, "snippets", f"{aid}.mp4")

        with c1:
            st.markdown("**Alert Snapshot**  <span class='badge'>Fused Score</span>  "
                        f"<span class='pill'>{sel['score']:.2f}</span>", unsafe_allow_html=True)
            if os.path.exists(frame_img):
                st.image(Image.open(frame_img), use_column_width=True, caption=os.path.basename(frame_img))
            else:
                st.info("No snapshot saved for this alert.")

        with c2:
            with st.expander("🎬 2-sec snippet", expanded=True):
                if os.path.exists(clip_mp4): st.video(clip_mp4)
                else: st.info("No snippet saved.")
            with st.expander("🧾 Raw record"):
                st.json(sel)

        st.markdown("---")
        explain_alert(sel, wS, wM, wL, wA)
        st.caption("This explainer estimates component contributions using the sliders above if per-component scores were not saved.")

# Gallery
with tab_gallery:
    st.subheader("Preview Frames")
    previews = sorted(glob.glob(os.path.join(out_dir, "frames", "preview_*.jpg")))
    if previews:
        per_row = st.slider("Thumbnails per row", 4, 12, 8)
        rows = (len(previews) + per_row - 1) // per_row
        for r in range(rows):
            cols = st.columns(per_row, gap="small")
            for j in range(per_row):
                k = r*per_row + j
                if k >= len(previews): break
                with cols[j]:
                    st.image(previews[k], use_column_width=True, output_format="JPEG")
    else:
        st.info("No preview frames generated yet (they save periodically during inference).")

# About
with tab_about:
    st.subheader("How this system works")
    st.markdown("""
    1. **Detection (YOLOv8 nano):** finds people & objects every frame.
    2. **Tracking:** follows detections across frames to measure dwell time.
    3. **Rules:**
       - **Loitering** — a person’s centroid stays inside a small radius for several seconds.
       - **Abandoned object** — a bag-like object remains without an overlapping person.
    4. **Motion model (3D Autoencoder):** learns normal walking from training clips and scores unusual motion.
    5. **Fusion:** `F = wS·Semantic + wM·Motion + wL·Loiter + wA·Abandon`.
       If `F` exceeds your threshold during inference, an alert is saved.
    """)
    st.markdown("""
    Use the **weights** in the sidebar to mirror your inference settings for accurate explanations.
    Export your filtered alerts with the button in the sidebar.
    """)
