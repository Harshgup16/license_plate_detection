<div align="center">

<br/>

```
██████╗ ██╗      █████╗ ████████╗███████╗    ███████╗ ██████╗ █████╗ ███╗   ██╗
██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝    ██╔════╝██╔════╝██╔══██╗████╗  ██║
██████╔╝██║     ███████║   ██║   █████╗      ███████╗██║     ███████║██╔██╗ ██║
██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝      ╚════██║██║     ██╔══██║██║╚██╗██║
██║     ███████╗██║  ██║   ██║   ███████╗    ███████║╚██████╗██║  ██║██║ ╚████║
╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝
```

### 🚗 Real-time Indian License Plate Detection & OCR — powered by YOLOv8 + EasyOCR

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7%2B-00C49A?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-F7B731?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-CPU%20%7C%20GPU-informational?style=for-the-badge)

<br/>

> Feed it a traffic video. Get back a fully annotated output with every license plate detected, cropped, read, and stamped — frame by frame.

<br/>

[**Features**](#-features) · [**How It Works**](#-how-it-works) · [**Installation**](#-installation) · [**Usage**](#-usage) · [**Training**](#-train-your-own-model) · [**Project Structure**](#-project-structure) · [**Troubleshooting**](#-troubleshooting)

---

</div>

<br/>

## 📸 What Does It Do?

```
  ┌──────────────────────────────────────────────────┐
  │          traffic_video.mp4  (input)               │
  │                                                  │
  │   ┌─────────────────────────────────┐            │
  │   │  ██████████████████████████████ │ ← zoomed   │
  │   │  ██  MH 12 AB 1234  ███████████ │   plate    │
  │   │  ██████████████████████████████ │   crop     │
  │   └─────────────────────────────────┘            │
  │                                                  │
  │   MH12AB1234   ← stable OCR overlay              │
  │   ┌──────────────────────┐                       │
  │   │                      │  ← bounding box       │
  │   │      [vehicle]       │                       │
  │   │                      │                       │
  │   └──────────────────────┘                       │
  └──────────────────────────────────────────────────┘
              ↓
  output_with_license.mp4  (annotated output)
```

<br/>

## ✨ Features

| | Feature | Details |
|---|---|---|
| 🔍 | **Plate Detection** | Fine-tuned YOLOv8n on ~30k Indian vehicle plate images |
| 🔤 | **Optical Character Recognition** | EasyOCR in English-only mode with Otsu thresholding + 2× upscaling |
| 🇮🇳 | **Format Auto-Correction** | Enforces `AA00AAA` Indian plate pattern; fixes digit/letter confusions |
| 🔁 | **Confusion Mapping** | Auto-corrects `0↔O`, `1↔I`, `5↔S`, `8↔B` based on position in plate |
| 📊 | **Stability Tracking** | Rolling 10-frame deque with majority-vote for rock-solid OCR results |
| 🎬 | **Full Video Pipeline** | Frame-by-frame processing with zoomed plate overlay in output |
| ⚡ | **Lightweight** | YOLOv8 **nano** — runs on CPU; GPU-optional for faster inference |

<br/>

## ⚙️ How It Works

The pipeline has four clean stages:

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   VIDEO FRAME                                                      │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────┐     conf ≥ 0.3                                  │
│  │  YOLOv8 nano │ ──────────────────► bounding boxes (x1,y1,x2,y2)│
│  └──────────────┘                                                  │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────────────────────┐                             │
│  │ OpenCV Preprocessing             │                             │
│  │  → Grayscale                     │                             │
│  │  → Otsu Thresholding             │                             │
│  │  → 2× Bicubic Upscale            │                             │
│  └──────────────────────────────────┘                             │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────────────────────┐                             │
│  │ EasyOCR                          │                             │
│  │  allowlist: A–Z  0–9             │                             │
│  └──────────────────────────────────┘                             │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────────────────────┐                             │
│  │ Format Corrector + Regex Check   │                             │
│  │  enforce AA00AAA pattern         │                             │
│  │  fix 0↔O, 1↔I, 5↔S, 8↔B         │                             │
│  │  validate ^[A-Z]{2}[0-9]{2}[A-Z]{3}$                          │
│  └──────────────────────────────────┘                             │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────────────────────┐                             │
│  │ Stability Tracker                │                             │
│  │  deque(maxlen=10) per box_id     │                             │
│  │  → most-frequent valid reading   │                             │
│  └──────────────────────────────────┘                             │
│       │                                                            │
│       ▼                                                            │
│   ANNOTATED FRAME  ──►  OUTPUT VIDEO                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 🇮🇳 Indian Plate Format

This project targets the standard **BH / State code** format used across India:

```
  M  H  1  2  A  B  1  2  3  4
  └──┘  └──┘  └──┘  └──────────┘
 State  RTO  Series    Number

  7-character short code:  MH12AB1  (AA + 00 + AAA)
```

<br/>

## 🛠 Installation

### 1 — Clone the Repository

```bash
git clone https://github.com/Harshgup16/license_plate_detection.git
cd license_plate_detection
```

### 2 — Create a Virtual Environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 <strong>requirements.txt</strong></summary>

```
ultralytics>=8.0.0
easyocr>=1.7.0
opencv-python>=4.8.0
numpy>=1.24.0
```

</details>

### 4 — Add Model Weights

Place your trained weights file at the project root:

```
license_plate_detection/
└── license_plate_best.pt   ← required
```

> No weights yet? See [Train Your Own Model](#-train-your-own-model) below.

<br/>

## 🚀 Usage

### Run the Detection Pipeline

```bash
python app.py
```

### Configure Input / Output

Open `app.py` and update these lines near the bottom of the file:

```python
input_video  = "input.mp4"              # ← path to your source video
output_video = "output_with_license.mp4"  # ← where to save the result
```

### Tune the Confidence Threshold

```python
CONF_THRESH = 0.3   # raise → fewer false positives (range: 0.0–1.0)
```

### Watch Progress

```
Video opened: 1920x1080 @ 30.0fps | Total frames: 450
Processing frame 142/450
Done! Output saved to: output_with_license.mp4
```

<br/>

## 🧠 Train Your Own Model

### Step 1 — Get a Dataset

Head to [roboflow.com](https://roboflow.com), find an **Indian License Plate** dataset, and export in **YOLOv8 format**.

### Step 2 — Train

```python
from ultralytics import YOLO
import shutil, os

model = YOLO("yolov8n.pt")   # start from pretrained nano weights

results = model.train(
    data=dataset.location + "/data.yaml",
    epochs=50,          # 50–100 recommended for good accuracy
    imgsz=640,
    batch=32,
    workers=4,
    device=0,           # 0 = first GPU  |  "cpu" for CPU-only
    cache=True,
    fraction=0.3        # use 30 % of dataset (~30k images)
)

# Save best weights
os.makedirs("saved_models", exist_ok=True)
shutil.copy(f"{results.save_dir}/weights/best.pt", "saved_models/license_plate_best.pt")
print("✅ Weights saved to saved_models/license_plate_best.pt")
```

### Step 3 — Use Your Weights

```python
# In app.py, update the model path:
model = YOLO("saved_models/license_plate_best.pt")
```

> The repo already ships with `license_plate_best.pt` — pre-trained on ~30k Indian plate images.

<br/>

## 📁 Project Structure

```
license_plate_detection/
│
├── app.py                        # 🔧 Main detection + OCR pipeline
├── number_plate_opencv.ipynb     # 📓 Training & experimentation notebook
├── license_plate_best.pt         # 🤖 Trained YOLOv8n weights
│
├── input.mp4                     # 🎬 Source video (you provide)
├── output_with_licensev3.mp4     # 🎥 Annotated output (auto-generated)
│
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # 📖 You are here
```

<br/>

## 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| `Error: Video file not found` | Update `input_video` path in `app.py` |
| OCR reads garbage characters | Motion blur / poor lighting degrades results; also try increasing `CONF_THRESH` |
| Very slow processing | Set `gpu=True` in `easyocr.Reader(...)` if CUDA is available |
| Wrong plate format / no match | The corrector assumes 7-char Indian plates; adapt `correct_plate_format()` for other regional formats |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your **activated** virtual environment |
| Output video won't play | Ensure `mp4v` codec is supported; try changing `fourcc` to `avc1` for H.264 |

<br/>

## 🤝 Contributing

Pull requests are welcome! For larger changes, please open an issue first.

```bash
# 1. Fork & clone
git checkout -b feature/your-improvement

# 2. Make changes, then commit
git commit -m "feat: describe your improvement"

# 3. Push and open a PR
git push origin feature/your-improvement
```

<br/>

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

<br/>

---

<div align="center">

Made with ❤️ for Indian roads

⭐ Star the repo if it helped you!

</div>
