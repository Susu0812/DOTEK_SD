"""Temporary three-record server for the root-agent Task 5 browser smoke."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from werkzeug.serving import run_simple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from low_light_dataset.review_service import create_app, find_available_port


RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="task5-browser-smoke-", dir=Path(__file__).parent))
CANDIDATE_ROOT = RUNTIME_ROOT / "candidates"
WORK_ROOT = RUNTIME_ROOT / "work"
FRAMES_ROOT = CANDIDATE_ROOT / "frames"
FRAMES_ROOT.mkdir(parents=True)

rows = []
preannotations = []
for index in range(3):
    stem = f"smoke_t{index:03d}"
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels[:, :, 0] = 24 + index * 18
    pixels[:, :, 1] = np.linspace(25, 115, 640, dtype=np.uint8)[None, :]
    pixels[:, :, 2] = np.linspace(35, 145, 480, dtype=np.uint8)[:, None]
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.line([(240 + 25 * index, 470), (280 + 10 * index, 330), (300, 170)], fill=(230, 40, 40), width=16)
    path = FRAMES_ROOT / f"{stem}.jpg"
    image.save(path, format="JPEG", quality=95)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(
        {
            "stem": stem,
            "source": "base" if index == 0 else "event",
            "event_reason": "one_second_base" if index == 0 else "motion_peak",
            "event_score": f"{index / 10:.6f}",
            "target_timestamp_seconds": f"{index:.6f}",
            "image_sha256": digest,
        }
    )
    preannotations.append(
        {
            "stem": stem,
            "target_timestamp_seconds": float(index),
            "image_sha256": digest,
            "anchors": [
                {"y": 170, "x": 300.0, "confidence": 0.7, "source": "fused"},
                {"y": 330, "x": 280.0 + 10 * index, "confidence": 0.8, "source": "fused"},
                {"y": 470, "x": 240.0 + 25 * index, "confidence": 0.9, "source": "fused"},
            ],
            "warnings": [],
            "source_metrics": {"final_count": 3},
        }
    )

with (CANDIDATE_ROOT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

preannotation_path = RUNTIME_ROOT / "preannotation.json"
preannotation_path.write_text(
    json.dumps({"schema_version": 1, "records": preannotations}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

app = create_app(CANDIDATE_ROOT, WORK_ROOT, preannotation_path=preannotation_path)
port = find_available_port()
print(f"SMOKE_URL=http://127.0.0.1:{port}/", flush=True)
print(f"SMOKE_RUNTIME={RUNTIME_ROOT}", flush=True)
run_simple("127.0.0.1", port, app, use_reloader=False, threaded=True)
