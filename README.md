# 🚗 LicenseLens — Indian License Plate Detection & Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7%2B-00C49A?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Real-time Indian vehicle license plate detection, OCR, and recognition from video streams — powered by YOLOv8 and EasyOCR.**

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Training](#-training-your-own-model) · [Usage](#-usage) · [How It Works](#-how-it-works) · [Project Structure](#-project-structure)

</div>

---

## 📸 Demo

> The system detects license plates in each video frame, zooms into the plate region, reads the text via OCR, and overlays the corrected plate number directly on the video.

```
Input:  traffic_video.mp4
Output: annotated video with bounding boxes + plate number overlay
        e.g.  ┌──────────────┐
              │  MH12AB1234  │  ← stable, corrected OCR result
              └──────────────┘
              [  plate crop  ]
                ┌──────────┐
                │ vehicle  │
                └──────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **Plate Detection** | Fine-tuned YOLOv8n on ~30k Indian plate images |
| 🔤 **OCR** | EasyOCR with English-only mode + Otsu thresholding + 2× upscaling |
| 🇮🇳 **Format Correction** | Enforces `AA00AAA` pattern; auto-corrects common digit/letter confusions (`0↔O`, `1↔I`, `5↔S`, `8↔B`) |
| 📊 **Stability Tracking** | Rolling window of 10 frames; picks the most-frequent reading per plate region |
| 🎬 **Video Pipeline** | Frame-by-frame processing with annotated output video |
| ⚡ **Lightweight** | YOLOv8 **nano** — runs on CPU; GPU optional |

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/licenselens.git
cd licenselens
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 <strong>requirements.txt</strong> (click to expand)</summary>

```txt
ultralytics>=8.0.0
easyocr>=1.7.0
opencv-python>=4.8.0
numpy>=1.24.0
```

</details>

### 4. Download / place the model weights

Place your trained weights file at the root of the project:

```
licenselens/
└── license_plate_best.pt   ← put it here
```

> Don't have weights yet? See [Training Your Own Model](#-training-your-own-model) below.

---

## 🚀 Usage

### Basic — process a video file

```bash
python detect.py
```

By default the script reads `input.mp4` from the current directory and writes the annotated result to the path configured in `output_video`.

### Configure input / output paths

Open `detect.py` and edit these two lines near the top of the *Video I/O* section:

```python
input_video  = "input.mp4"          # ← your source video
output_video = "output_plates.mp4"  # ← where to save results
```

### Tune the confidence threshold

```python
CONF_THRESH = 0.3   # raise to reduce false positives (0–1)
```

### Run and watch progress

```
Video opened: 1920x1080 @ 30.0fps | Total frames: 450
Processing frame 142/450
```

The annotated video is saved automatically when processing finishes.

---

## 🧠 Training Your Own Model

### 1. Get the dataset (Roboflow)

Sign in at [roboflow.com](https://roboflow.com), find an Indian license plate dataset, and export it in **YOLOv8 format**. The training snippet below uses the Roboflow `dataset.location` variable directly.

### 2. Train

```python
from ultralytics import YOLO
import shutil, os

model = YOLO("yolov8n.pt")   # start from pretrained nano weights

results = model.train(
    data=dataset.location + "/data.yaml",
    epochs=1,          # increase for better accuracy (50–100 recommended)
    imgsz=640,
    batch=32,
    workers=4,
    device=0,          # 0 = first GPU; "cpu" for CPU-only
    cache=True,
    fraction=0.3       # use 30% of dataset (~30k images) — adjust as needed
)

# Save the best weights
os.makedirs("saved_models", exist_ok=True)
shutil.copy(str(results.save_dir) + "/weights/best.pt",
            "saved_models/license_plate_best.pt")
shutil.copy(str(results.save_dir) + "/weights/last.pt",
            "saved_models/license_plate_last.pt")

print("✅ Weights saved in saved_models/")
```

### 3. Use your trained weights

Copy `saved_models/license_plate_best.pt` to the project root and rename it `license_plate_best.pt`, or update the path in `detect.py`:

```python
model = YOLO("saved_models/license_plate_best.pt")
```

---

## ⚙️ How It Works

```
Video Frame
    │
    ▼
┌─────────────────────┐
│   YOLOv8 Detection  │  → bounding boxes (x1,y1,x2,y2) + confidence
└─────────────────────┘
    │  conf ≥ 0.3
    ▼
┌─────────────────────┐
│   Plate Crop + Pre- │  grayscale → Otsu threshold → 2× upscale
│   processing (OpenCV│
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   EasyOCR           │  allowlist: A-Z 0-9
└─────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Format Correction       │  enforce AA00AAA; fix 0↔O, 1↔I, 5↔S, 8↔B
│  + Regex Validation      │  ^[A-Z]{2}[0-9]{2}[A-Z]{3}$
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Stability Tracker       │  rolling deque(maxlen=10) per plate region
│  (deque + majority vote) │  → most-frequent valid reading
└──────────────────────────┘
    │
    ▼
  Annotated Frame → Output Video
```

### Plate format assumption

This project targets the standard **Indian vehicle registration** format:

```
  R  A  J  0  1  A  B  1  2  3  4
  └──┘  └──┘  └──┘  └──────────┘
  State  RTO   Series   Number
```

The 7-character short code captured here follows the pattern **`AA00AAA`** (two letters, two digits, three letters).

---

## 📁 Project Structure

```
licenselens/
├── detect.py                  # main detection + OCR pipeline
├── license_plate_best.pt      # trained YOLOv8 weights (you provide)
├── input.mp4                  # source video (you provide)
├── output_with_licensev3.mp4  # annotated output (generated)
├── requirements.txt
└── README.md
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Error: Video file not found` | Check the `input_video` path in `detect.py` |
| OCR reads garbage text | Lower lighting / motion blur degrades results; try raising `CONF_THRESH` |
| Very slow processing | Switch `gpu=False` → `gpu=True` in `easyocr.Reader` if CUDA is available |
| Wrong plate format | The corrector assumes **7-character Indian plates**; adapt `correct_plate_format()` for other formats |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your activated virtual environment |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/improve-ocr`
3. Commit your changes: `git commit -m 'feat: improve OCR preprocessing'`
4. Push to the branch: `git push origin feature/improve-ocr`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

Made with ❤️ for Indian roads · Star ⭐ the repo if it helped you!

</div>
