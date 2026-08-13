# 🎥 Smart CCTV Analytics

An AI-powered video surveillance system that detects and tracks people and vehicles in CCTV footage, then automatically flags security-relevant events — **loitering**, **crowding**, and **restricted-zone intrusion** — in real time through an interactive dashboard.

Built with **YOLO11** for object detection and **ByteTrack** for multi-object tracking, wrapped in a **Streamlit** dashboard for live monitoring and reporting.

---

## ✨ Features

- **Real-time detection & tracking** of people and vehicles (car, truck, bus, motorcycle) using YOLO11 + ByteTrack
- **Smart event detection**:
  - 🚷 **Loitering** — flags a person who stays almost stationary for longer than a configurable time threshold
  - 👥 **Crowding** — flags when the number of people in frame exceeds a configurable threshold
  - ⛔ **Restricted-zone intrusion** — flags when a person enters a user-defined restricted polygon zone
  - A **safe waiting zone** (e.g. a queue) is exempt from loitering checks
- **Interactive Streamlit dashboard**:
  - Upload a video or pick from sample videos
  - Live annotated video feed with drawn zones and bounding boxes
  - Real-time KPI cards (total people, total vehicles, average dwell time, total alerts)
  - Live security events log
  - Export the annotated video and a CSV events report after processing
- **Configurable via a single `config.yaml`** — thresholds, zones, model, and performance settings are all in one place, no code changes needed
- **Clean, layered architecture** (`core` / `infrastructure` / `presentation`) that keeps the detection logic, I/O utilities, and UI cleanly separated

---

## 🏗️ Architecture

```
Smart-CCTV-Analytics/
├── core/                     # Detection, tracking & analytics logic
│   ├── base_detector.py      # Abstract interface all detectors implement
│   ├── detection_model.py    # YOLO11 + ByteTrack detector implementation
│   ├── tracker.py            # Converts raw YOLO results into TrackedObject instances
│   └── analytics.py          # Event detection engine (loitering, crowding, restricted zone)
├── infrastructure/           # Cross-cutting utilities
│   ├── config_loader.py      # Loads and validates config.yaml
│   ├── logger.py             # Unified logging setup
│   └── video_io.py           # Video reading/writing helpers
├── presentation/
│   └── app.py                # Streamlit dashboard (entry point)
├── videos/                   # Sample input videos (place your own here)
├── outputs/                  # Annotated output videos are saved here
├── config.yaml                # All tunable settings
├── requirements.txt
├── yolo11n.pt / yolo11s.pt   # YOLO11 model weights
└── .devcontainer/            # Ready-to-run GitHub Codespaces / VS Code Dev Container setup
```

**How a frame flows through the system:**

1. `DetectionModel` runs YOLO11 detection + ByteTrack tracking on the frame (`core/detection_model.py`)
2. `Tracker` converts the raw YOLO results into clean `TrackedObject` instances (`core/tracker.py`)
3. `AnalyticsEngine` analyzes the tracked objects against the configured zones and thresholds, and emits `Event` objects for anything noteworthy (`core/analytics.py`)
4. `presentation/app.py` renders the annotated frame, updates the live KPIs, and logs events to the dashboard

The `BaseDetector` interface (`core/base_detector.py`) means new detector types (e.g. a segmentation-based model) can be added later without touching the tracking, analytics, or UI layers.

---

## ⚙️ Configuration

All behavior is controlled from `config.yaml`:

```yaml
model:
  weights: "yolo11s.pt"       # YOLO model to use
  confidence: 0.45            # Detection confidence threshold
  iou: 0.50                   # IoU threshold for NMS
  classes: [0, 2, 3, 5, 7]    # COCO class IDs: person, car, motorcycle, bus, truck
  device: "cpu"                # "cpu" or "cuda"

tracker:
  type: "bytetrack.yaml"
  track_buffer: 60

events:
  loitering_seconds: 20        # Seconds stationary before a loitering alert
  crowd_threshold: 15          # People count that triggers a crowding alert
  restricted_zone: [...]       # Polygon points [x, y] defining the restricted area
  safe_waiting_zone: [...]     # Polygon points [x, y] exempt from loitering checks

performance:
  frame_resize_width: 640
  segmentation_every_n_frames: 5

output:
  save_annotated_video: true
  report_format: "both"
```

> The restricted zone and safe waiting zone are defined as pixel-coordinate polygons matched to your video's resolution — adjust the points to match your camera's field of view.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Jana-Sherif-Mohamed/Smart-CCTV-Analytics
cd Smart-CCTV-Analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add a video

Place a sample video (`.mp4`, `.avi`, `.mov`, or `.mkv`) inside the `videos/` folder, or plan to upload one directly from the dashboard.

### 4. Run the dashboard

Then open the URL Streamlit prints (`https://smart-cctv-analytics-2xhdxfzcftcedtg7qe5jud.streamlit.app/`) in your browser.

> 💡 The project also includes a ready-to-use `.devcontainer` setup — opening it in GitHub Codespaces or VS Code will install dependencies and launch the dashboard automatically.

### 5. Use the dashboard

1. Choose a sample video or upload your own from the sidebar
2. Pick a YOLO model (`yolo11n.pt` for speed, `yolo11s.pt` for better accuracy) and adjust the confidence threshold
3. Click **▶️ Start Analytics**
4. Watch the live annotated feed, KPIs, and events log update in real time
5. Once processing finishes, download the annotated video and/or the CSV events report

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [YOLO11](https://docs.ultralytics.com/) (Ultralytics) |
| Multi-Object Tracking | ByteTrack |
| Dashboard / UI | [Streamlit](https://streamlit.io/) |
| Video Processing | OpenCV |
| Config | PyYAML |

---

## 📊 Output

After each run, the system produces:

- An **annotated video** (`outputs/`) with bounding boxes, track IDs, and drawn zones
- A **CSV events report** with the timestamp, event type, track ID, and message for every detected event
- Live **summary statistics**: total people, total vehicles, average dwell time, and alert counts broken down by type

---

## 🗺️ Roadmap

- [ ] Segmentation-based detection mode (interface already supports it via `BaseDetector`)
- [ ] Multi-camera / multi-stream support
- [ ] Persistent database storage for historical event analysis
- [ ] Configurable alert notifications (email/Slack/webhook)

---

## 📄 License

This project is developed for academic purposes as part of a graduation project.
