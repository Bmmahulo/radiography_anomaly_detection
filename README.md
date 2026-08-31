---
title: Radiography Anomaly Detection
emoji: 🩻
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Radiography Anomaly Detection — Local Prototype

An AI-assisted triage prototype that analyzes X-ray scans, flags likely
anomalies (fractures, thoracic lesions, etc.), and returns a Grad-CAM
heatmap so a reviewing clinician can quickly verify *why* a scan was
flagged. Built with **PyTorch**, **FastAPI**, **OpenCV**, and
**Albumentations**.

> ⚠️ **Not a medical device.** This is a research/engineering prototype for
> screening-assistance workflows. It has not been validated, cleared, or
> approved for standalone diagnostic use. Every prediction requires
> physician review before any clinical decision is made.

---

## 1. Project Structure

```
radiography-anomaly-detection/
├── app.py                      # FastAPI application entrypoint
├── train.py                    # Training script
├── inference.py                # Single-image inference + Grad-CAM CLI
├── config.py                   # Central configuration (paths, hyperparams)
├── requirements.txt
├── api/
│   ├── routes.py                # /api/v1/scan/analyze, /api/v1/health
│   └── schemas.py                # Pydantic request/response models
├── models/
│   ├── classifier.py             # EfficientNet/ResNet backbone + multi-label head
│   └── gradcam.py                 # Grad-CAM implementation
├── data/
│   ├── dataset.py                 # DICOM/PNG/JPEG-compatible Dataset class
│   └── transforms.py              # Albumentations train/val pipelines
├── utils/
│   ├── logger.py                   # Centralized logging
│   └── metrics.py                  # AUROC / Precision / Recall / F1
├── scripts/
│   └── make_sample_dataset.py       # Generates a synthetic dataset for smoke-testing
├── static/heatmaps/                 # Grad-CAM overlays saved here, served via /static
├── checkpoints/                     # Trained model weights (best_model.pt, last_model.pt)
├── dataset_sample/                  # Default location for train/val CSVs + images
└── logs/                            # Rotating file logs (app.log)
```

---

## 2. Setup (Ubuntu Linux, Python 3.10+)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# If you have an NVIDIA GPU, install the CUDA build of torch instead of the
# CPU-only default that requirements.txt may resolve to, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

System dependencies (Ubuntu):
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```
(`libgl1`/`libglib2.0-0` are required by OpenCV's `cv2.imread`/`cv2.imwrite`.)

---

## 3. Dataset Integration

The pipeline expects **two CSV files** (train + validation), each with:

| column        | description                                    |
|---------------|-------------------------------------------------|
| `image_path`  | filename or absolute path to the scan            |
| one column per class in `config.CLASS_NAMES` (0/1) | multi-label ground truth |

Default class list (edit in `config.py` to match your dataset):
```python
CLASS_NAMES = ["fracture", "lesion", "pneumonia", "pneumothorax", "cardiomegaly", "nodule"]
```

### Option A — NIH ChestX-ray14
Download from the [NIH Clinical Center release](https://nihcc.app.box.com/v/ChestXray-NIHCC).
The provided `Data_Entry_2017.csv` has an `Image Index` column and a
pipe-separated `Finding Labels` column. Convert it into the expected
one-hot format with a short pandas script (multi-hot encode `Finding Labels`,
rename `Image Index` → `image_path`), then point `--train-csv`/`--val-csv`
at the resulting files.

### Option B — FracAtlas (fracture detection)
FracAtlas ships with a `fractured` binary column per image. Map it onto a
single-class version of `config.CLASS_NAMES = ["fracture"]` and adjust
`config.NUM_CLASSES` accordingly.

### Option C — DICOM studies (PACS export)
`data/dataset.py::load_image` reads `.dcm` files directly via `pydicom`,
normalizing pixel data to 8-bit RGB automatically — just reference `.dcm`
paths in your CSV's `image_path` column.

### Quick smoke test (no real dataset required)
```bash
python scripts/make_sample_dataset.py --num-train 40 --num-val 10
```
This generates synthetic images + random labels under `dataset_sample/` so
you can verify the full pipeline runs end-to-end before plugging in real
data.

### Real dataset: FracAtlas (recommended starting point on CPU)
Single-label fracture detection, ~4,000 images, ~323MB — small enough to
fully train on CPU in a reasonable time.

1. Download `FracAtlas.zip` (no registration required):
   https://doi.org/10.6084/m9.figshare.22363012
2. Extract it somewhere, e.g. `C:\datasets\FracAtlas\`
3. Run:
   ```bash
   python scripts/prepare_fracatlas.py --source-dir /path/to/FracAtlas
   ```
4. Update `config.py`:
   ```python
   CLASS_NAMES = ["fracture"]
   ```
5. Train:
   ```bash
   python train.py --epochs 10 --backbone resnet34
   ```

### Real dataset: NIH ChestX-ray14 (multi-finding, larger)
14 thoracic findings, ~112,000 images total (~42GB) — the script works
against however many images you've actually downloaded, so you can start
with a small slice.

1. Download the metadata CSV (`Data_Entry_2017_v2020.csv`) and one or two
   `images_0XX.zip` archives (~2GB / ~10,000 images each) from:
   https://nihcc.app.box.com/v/ChestXray-NIHCC
   (or the single-download Kaggle mirror: https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Extract all image archives into one flat folder.
3. Run:
   ```bash
   python scripts/prepare_nih_chestxray.py \
       --csv-path /path/to/Data_Entry_2017_v2020.csv \
       --images-dir /path/to/images \
       --max-images 3000
   ```
4. Update `config.py` with the class list the script prints out, e.g.:
   ```python
   CLASS_NAMES = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
                  "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                  "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
   ```
5. Train:
   ```bash
   python train.py --epochs 10 --imbalance-strategy weighted_loss
   ```

Both scripts write absolute image paths into the train/val CSVs, so your
downloaded dataset never gets copied or duplicated — it's referenced in
place.

---

## 4. Training

```bash
python train.py \
  --train-csv dataset_sample/train_labels.csv \
  --val-csv dataset_sample/val_labels.csv \
  --backbone efficientnet_b0 \
  --epochs 25 \
  --batch-size 16 \
  --imbalance-strategy weighted_loss
```

Key features:
- **Backbones**: `efficientnet_b0`, `efficientnet_b3`, `resnet50`, `resnet34`
  (ImageNet-pretrained via `torchvision.models`, final layer replaced with a
  multi-label linear head).
- **Class imbalance handling** (`--imbalance-strategy`):
  - `weighted_loss` — `pos_weight` in `BCEWithLogitsLoss`, computed from
    train-split class frequencies.
  - `sampler` — `WeightedRandomSampler` up-weighting rare-finding samples.
  - `both` — combine both strategies.
  - `none` — plain BCE, uniform sampling.
- **Metrics** (`utils/metrics.py`): per-class + macro-averaged AUROC,
  Precision, Recall, F1 — computed every epoch on the validation split.
- **Checkpointing**: `checkpoints/best_model.pt` (lowest val loss) and
  `checkpoints/last_model.pt` (most recent epoch), each storing model
  weights + class names + backbone name for reproducible loading.
- **Early stopping** on validation loss plateau (`--patience`).
- Mixed-precision training automatically enabled on CUDA devices.

---

## 5. Inference (CLI)

```bash
python inference.py --image /path/to/scan.png
python inference.py --image /path/to/study.dcm --checkpoint checkpoints/best_model.pt
```

Outputs a JSON result to stdout and saves an annotated Grad-CAM overlay to
`static/heatmaps/`:

```json
{
  "image_path": "/path/to/scan.png",
  "priority": "HIGH",
  "top_finding": "fracture",
  "top_finding_confidence": 0.8734,
  "flagged_findings": ["fracture", "lesion"],
  "class_scores": {
    "fracture": 0.8734, "lesion": 0.612, "pneumonia": 0.041,
    "pneumothorax": 0.018, "cardiomegaly": 0.09, "nodule": 0.221
  },
  "heatmap_url": "/static/heatmaps/heatmap_a1b2c3d4e5f6.png"
}
```

Triage priority thresholds (tunable in `config.py`):
- `HIGH` — top finding confidence ≥ 0.75
- `MEDIUM` — top finding confidence ≥ 0.5
- `LOW` — below 0.5

---

## 6. Running the REST API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: **http://localhost:8000/docs**

### `POST /api/v1/scan/analyze`
Multipart file upload → JSON prediction + heatmap URL.

```bash
curl -X POST "http://localhost:8000/api/v1/scan/analyze" \
  -F "file=@/path/to/scan.png"
```

Response matches the `ScanAnalysisResponse` schema (`api/schemas.py`):
`request_id`, `priority`, `top_finding`, `top_finding_confidence`,
`flagged_findings`, `class_scores`, `heatmap_url`, `processing_time_ms`,
`model_version`.

Heatmap images are served statically at
`http://localhost:8000/static/heatmaps/<filename>.png`.

## 7. Friendly Next.js Frontend

The `frontend/` directory contains a responsive Next.js + Tailwind interface
for uploading scans, checking API health, and reviewing predictions. Start
the FastAPI service first, then run the frontend:

```bash
cd frontend
npm install
```

Create `frontend/.env.local` when the API is not running on the default local
address:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Then launch the UI:

```bash
npm run dev
```

Open **http://localhost:3000**. For a separate Render frontend service, set
`NEXT_PUBLIC_API_URL` to the public URL of the FastAPI Render service and use
`npm run build && npm run start` as the build/start workflow.

### `GET /api/v1/health`
Reports whether the model checkpoint loaded successfully, current device
(CPU/GPU), and backbone in use — useful for container/orchestration health
probes.

---

## 8. Error Handling Notes

- Corrupt/unreadable images return **HTTP 422** with a descriptive message
  rather than crashing the worker.
- Unsupported file extensions return **HTTP 415**.
- Oversized uploads (> `config.MAX_UPLOAD_SIZE_MB`, default 25MB) are
  rejected mid-stream with **HTTP 413** before being fully buffered.
- If Grad-CAM generation fails for some reason, the classification result is
  still returned (with a `heatmap_error` field) rather than failing the
  whole request — a partial result beats no result in a triage context.
- All requests are logged with a short `request_id` for traceability
  (`logs/app.log`, rotated at 5MB).

---

## 9. Version Control (Git)

```bash
cd radiography-anomaly-detection
git init
git add .
git commit -m "Initial radiography anomaly detection prototype"
```
Create an empty repo on GitHub (no README/gitignore selected there), then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
The included `.gitignore` already excludes `venv/`, model checkpoints,
generated heatmaps, and logs, so large/generated files won't get committed
by accident.

## 10. Opening in VS Code

From inside the project folder:
```bash
code .
```
(Requires VS Code's `code` CLI on PATH — if it's not recognized, open VS
Code manually via File → Open Folder instead.) From there, GitHub Copilot
or any other in-editor assistant has full access to the codebase for
further iteration.

## 11. Extending This Prototype

- Swap `config.CLASS_NAMES` + `config.NUM_CLASSES` to match your dataset's
  actual finding taxonomy.
- Add a persistence layer (e.g. Postgres) to store analysis history per
  patient/study instead of the current stateless request/response model.
- Add authentication/authorization (e.g. OAuth2 + hospital SSO) before any
  non-local deployment — the current CORS config (`allow_origins=["*"]`) is
  intentionally permissive for local prototyping only.
- Add DICOM metadata extraction (patient ID, study date, modality) alongside
  pixel data for richer triage context.
- Consider an async task queue (Celery/RQ) if scan volume requires decoupling
  upload from inference latency.
