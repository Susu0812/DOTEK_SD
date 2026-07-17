# Low-Light Labeled Training Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract 60 frames from the first five minutes of `camera_full_rgb.mp4`, enhance them, produce reviewed binary hose-centerline labels, and atomically merge the 60 valid pairs into the existing training set without starting training.

**Architecture:** A focused `low_light_dataset` package separates frame/enhancement operations, model-assisted annotation geometry, artifact generation, and transactional dataset merging. A CLI exposes `prepare`, `finalize`, `validate`, and `merge` phases so generated labels can be visually reviewed before the formal training set is changed.

**Tech Stack:** Python 3.10, NumPy 1.24, OpenCV 4.9, Pillow 10.4, SciPy 1.10, PyTorch 2.1, torchvision 0.16, built-in `unittest`, CSV/JSON/SHA-256 from the standard library.

## Global Constraints

- Input video: `E:\New_DOTEK\数据\camera_full_rgb.mp4`, process `[0, 300)` seconds at exactly 5-second intervals.
- Output exactly 60 enhanced JPG images and 60 same-stem PNG labels at 640×480.
- JPG images are 8-bit three-channel with quality 95; labels are single-channel with values only 0 and 1.
- Use `logs/0209_1509_lr_1e-04_b_64/best_model.pth`, model shape `(51, 18, 1)`, input `1×3×288×384`.
- Merge only enhanced images plus reviewed final labels; preserve originals and diagnostic artifacts outside `newdata/train`.
- Training must grow from 4858/4858 to 4918/4918; test must remain 867/867.
- Do not run training, fine-tuning, evaluation, or model export.
- The workspace is not a valid Git repository; replace commit steps with explicit status checkpoints and do not initialize Git.

---

## File Structure

- Create `low_light_dataset/__init__.py`: stable package exports.
- Create `low_light_dataset/image_ops.py`: timestamps, video extraction, quality metrics, adaptive Retinex-LAB enhancement, Unicode-safe image saving.
- Create `low_light_dataset/annotation.py`: model loading/inference, logits decoding, dual-prediction fusion, anchor regularization, dense centerline rasterization.
- Create `low_light_dataset/artifacts.py`: manifests, annotation JSON, overlays, contact sheets, final bundle validation.
- Create `low_light_dataset/dataset_merge.py`: snapshots, leakage/collision checks, transactional merge, receipt generation.
- Create `scripts/build_low_light_training_dataset.py`: phase-based command-line orchestration, baseline snapshot, and final verification.
- Create `tests/test_low_light_image_ops.py`: timestamp and enhancement tests.
- Create `tests/test_low_light_annotation.py`: synthetic logits, fusion, interpolation, label-format tests.
- Create `tests/test_low_light_artifacts.py`: file-pair and review-gate tests.
- Create `tests/test_low_light_dataset_merge.py`: collision, leakage, rollback, and successful merge tests.
- Create generated `datasets/low_light_camera_full_rgb/`: original/enhanced/initial/final labels, overlays, manifests, explicit review decisions, and merge receipt.

---

### Task 1: Frame Sampling and Adaptive Enhancement

**Files:**
- Create: `low_light_dataset/__init__.py`
- Create: `low_light_dataset/image_ops.py`
- Test: `tests/test_low_light_image_ops.py`

**Interfaces:**
- Produces: `sample_times(duration_seconds: int = 300, interval_seconds: int = 5) -> tuple[int, ...]`
- Produces: `FrameMetrics`, `EnhancementParams`, and `EnhancedFrame` frozen dataclasses.
- Produces: `measure_frame(frame_bgr: np.ndarray) -> FrameMetrics`
- Produces: `enhance_low_light(frame_bgr: np.ndarray) -> EnhancedFrame`
- Produces: `read_video_frame(capture: cv2.VideoCapture, timestamp_seconds: int) -> tuple[np.ndarray, float]`
- Produces: `save_jpeg(path: Path, frame_bgr: np.ndarray, quality: int = 95) -> None`

- [ ] **Step 1: Write failing timestamp and image-contract tests**

```python
class TimestampTests(unittest.TestCase):
    def test_first_five_minutes_every_five_seconds(self):
        values = sample_times()
        self.assertEqual(len(values), 60)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 295)
        self.assertTrue(all(b - a == 5 for a, b in zip(values, values[1:])))

class EnhancementTests(unittest.TestCase):
    def test_dark_frame_is_brightened_without_changing_contract(self):
        image = np.full((480, 640, 3), 35, dtype=np.uint8)
        result = enhance_low_light(image)
        self.assertEqual(result.image.shape, image.shape)
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertGreater(result.after.median, result.before.median)
        self.assertLessEqual(result.after.highlight_fraction, 0.01)

    def test_bright_frame_is_not_overexposed(self):
        ramp = np.tile(np.linspace(90, 220, 640, dtype=np.uint8), (480, 1))
        image = np.repeat(ramp[:, :, None], 3, axis=2)
        result = enhance_low_light(image)
        self.assertLess(result.params.retinex_weight, 0.25)
        self.assertLess(result.after.highlight_fraction, 0.02)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_image_ops -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'low_light_dataset'`.

- [ ] **Step 3: Implement timestamp generation, metric dataclasses, and input validation**

```python
@dataclass(frozen=True)
class FrameMetrics:
    mean: float
    median: float
    p05: float
    p95: float
    dark_fraction: float
    highlight_fraction: float
    laplacian_variance: float

def sample_times(duration_seconds=300, interval_seconds=5):
    if duration_seconds <= 0 or interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    return tuple(range(0, duration_seconds, interval_seconds))

def validate_bgr(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")

def measure_frame(frame):
    validate_bgr(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FrameMetrics(float(gray.mean()), float(np.median(gray)),
                        float(np.percentile(gray, 5)), float(np.percentile(gray, 95)),
                        float((gray < 40).mean()), float((gray > 245).mean()),
                        float(cv2.Laplacian(gray, cv2.CV_64F).var()))

def robust_uint8(values, low, high):
    lo, hi = np.percentile(values, (low, high))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

def limited_gray_world(frame, gain_min, gain_max):
    means = frame.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1.0), gain_min, gain_max)
    return np.clip(frame.astype(np.float32) * gains, 0, 255).astype(np.uint8)

def protect_highlights(original_l, candidate_l):
    mask = np.clip((original_l.astype(np.float32) - 180.0) / 65.0, 0.0, 1.0)
    protected = candidate_l.astype(np.float32) * (1.0 - mask) + original_l.astype(np.float32) * mask
    return np.clip(protected, 0, 255).astype(np.uint8)

def conditional_unsharp(frame, dark_fraction, laplacian_variance):
    amount = np.clip(0.45 - 0.35 * dark_fraction, 0.10, 0.45)
    if laplacian_variance > 500:
        amount *= 0.5
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
```

- [ ] **Step 4: Implement adaptive Retinex-LAB enhancement**

```python
def enhance_low_light(frame_bgr):
    validate_bgr(frame_bgr)
    before = measure_frame(frame_bgr)
    darkness = np.clip((115.0 - before.median) / 90.0, 0.0, 1.0)
    retinex_weight = float(0.08 + 0.52 * darkness)
    gamma = float(np.clip(np.log(115.0 / 255.0) /
                          np.log(max(before.median, 1.0) / 255.0), 0.55, 1.0))
    balanced = limited_gray_world(frame_bgr, gain_min=0.85, gain_max=1.18)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    denoised = cv2.bilateralFilter(l, 7, 24 + int(20 * darkness), 35)
    illumination = cv2.GaussianBlur(denoised, (0, 0), sigmaX=31)
    reflectance = np.log1p(denoised.astype(np.float32)) - np.log1p(illumination.astype(np.float32))
    retinex = robust_uint8(reflectance, low=1.0, high=99.0)
    restored = cv2.addWeighted(denoised, 1.0 - retinex_weight, retinex, retinex_weight, 0)
    gamma_lut = np.clip((np.arange(256) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    toned = cv2.LUT(restored, gamma_lut)
    clahe = cv2.createCLAHE(clipLimit=1.6 + 0.8 * darkness, tileGridSize=(8, 8)).apply(toned)
    final_l = protect_highlights(l, cv2.addWeighted(toned, 0.65, clahe, 0.35, 0))
    output = cv2.cvtColor(cv2.merge((final_l, a, b)), cv2.COLOR_LAB2BGR)
    output = conditional_unsharp(output, before.dark_fraction, before.laplacian_variance)
    params = EnhancementParams(gamma, retinex_weight, 1.6 + 0.8 * darkness)
    return EnhancedFrame(output, before, measure_frame(output), params)
```

- [ ] **Step 5: Implement precise video seeking and Unicode-safe JPEG writing**

```python
def read_video_frame(capture, timestamp_seconds):
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame at {timestamp_seconds}s")
    actual_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
    return frame, actual_ms / 1000.0

def save_jpeg(path, frame_bgr, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, format="JPEG", quality=quality, subsampling=0)
```

- [ ] **Step 6: Run Task 1 tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_image_ops -v`  
Expected: all Task 1 tests PASS.

- [ ] **Step 7: Record checkpoint**

Run: `Get-FileHash low_light_dataset\image_ops.py; Get-FileHash tests\test_low_light_image_ops.py`  
Expected: two SHA-256 values recorded in the task log; no Git commit because the workspace is not a repository.

---

### Task 2: Model-Assisted Centerline Annotation

**Files:**
- Create: `low_light_dataset/annotation.py`
- Test: `tests/test_low_light_annotation.py`

**Interfaces:**
- Consumes: BGR `uint8` frames from Task 1.
- Produces: `AnchorPrediction(y: int, x: float, confidence: float, source: str)`.
- Produces: `HoseAnnotator(checkpoint_path: Path, device: str = "cuda")` and `predict(frame_bgr) -> list[AnchorPrediction]`.
- Produces: `decode_logits(logits: np.ndarray, width: int, height: int) -> list[AnchorPrediction]`.
- Produces: `fuse_predictions(original, enhanced, max_disagreement_px=48) -> tuple[list[AnchorPrediction], list[str]]`.
- Produces: `regularize_anchors(predictions) -> tuple[list[AnchorPrediction], list[str]]`.
- Produces: `rasterize_centerline(anchors, width=640, height=480) -> np.ndarray`.

- [ ] **Step 1: Write failing geometry and label-format tests**

```python
class AnnotationGeometryTests(unittest.TestCase):
    def test_rasterized_label_is_binary_and_at_most_one_pixel_per_row(self):
        anchors = [AnchorPrediction(220, 100, .9, "fused"),
                   AnchorPrediction(300, 180, .9, "fused"),
                   AnchorPrediction(440, 260, .9, "fused")]
        label = rasterize_centerline(anchors)
        self.assertEqual(label.shape, (480, 640))
        self.assertEqual(label.dtype, np.uint8)
        self.assertEqual(set(np.unique(label)), {0, 1})
        self.assertLessEqual(int(label.sum(axis=1).max()), 1)

    def test_fusion_prefers_agreeing_high_confidence_points(self):
        original = [AnchorPrediction(300, 100, .7, "original")]
        enhanced = [AnchorPrediction(300, 110, .9, "enhanced")]
        fused, warnings = fuse_predictions(original, enhanced)
        self.assertAlmostEqual(fused[0].x, 105.625, places=3)
        self.assertEqual(warnings, [])

    def test_outlier_is_removed(self):
        values = [AnchorPrediction(220, 100, .9, "fused"),
                  AnchorPrediction(240, 110, .9, "fused"),
                  AnchorPrediction(260, 500, .8, "fused"),
                  AnchorPrediction(280, 130, .9, "fused")]
        cleaned, warnings = regularize_anchors(values)
        self.assertNotIn(500, [round(item.x) for item in cleaned])
        self.assertIn("removed_outlier", warnings)
```

- [ ] **Step 2: Run tests and verify missing annotation module failure**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_annotation -v`  
Expected: FAIL importing `low_light_dataset.annotation`.

- [ ] **Step 3: Implement checkpoint loading and exact project preprocessing**

```python
class HoseAnnotator:
    def __init__(self, checkpoint_path, device="cuda"):
        requested = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.device = requested
        self.model = parsingNet(pretrained=False, cls_dim=(51, 18, 1), use_aux=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
        incompatible = self.model.load_state_dict(state, strict=False)
        if any(not key.startswith("aux_") for key in incompatible.unexpected_keys):
            raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
        self.model.to(self.device).eval()

    def predict(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (384, 288), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().div(255.0)[None, None].repeat(1, 3, 1, 1)
        mean = torch.tensor((.485, .456, .406))[None, :, None, None]
        std = torch.tensor((.229, .224, .225))[None, :, None, None]
        with torch.no_grad():
            logits = self.model(((tensor - mean) / std).to(self.device)).cpu().numpy()
        return decode_logits(logits, frame_bgr.shape[1], frame_bgr.shape[0])
```

- [ ] **Step 4: Decode 51-class logits into confident image coordinates**

```python
ROW_ANCHORS = np.asarray([121, 131, 141, 150, 160, 170, 180, 189, 199,
                          209, 219, 228, 238, 248, 258, 267, 277, 287])

def decode_logits(logits, width, height, min_confidence=0.35):
    if logits.shape != (1, 51, 18, 1):
        raise ValueError(f"expected (1, 51, 18, 1), got {logits.shape}")
    full_prob = scipy.special.softmax(logits[0, :, :, 0], axis=0)
    foreground = scipy.special.softmax(logits[0, :50, :, 0], axis=0)
    expected = (foreground * np.arange(1, 51)[:, None]).sum(axis=0)
    result = []
    for index, row in enumerate(ROW_ANCHORS):
        foreground_confidence = 1.0 - float(full_prob[50, index])
        if foreground_confidence < min_confidence:
            continue
        x = (expected[index] - 1.0) * (width - 1.0) / 49.0
        y = int(round(row * (height - 1.0) / 287.0))
        result.append(AnchorPrediction(y, float(np.clip(x, 0, width - 1)),
                                       foreground_confidence, "model"))
    return result
```

- [ ] **Step 5: Implement confidence fusion, outlier rejection, and dense rasterization**

```python
def rasterize_centerline(anchors, width=640, height=480):
    if len(anchors) < 3:
        raise ValueError("at least three valid anchors are required")
    ordered = sorted(anchors, key=lambda item: item.y)
    ys = np.asarray([item.y for item in ordered], dtype=np.float64)
    xs = np.asarray([item.x for item in ordered], dtype=np.float64)
    dense_y = np.arange(max(0, int(np.ceil(ys[0]))),
                        min(height - 1, int(np.floor(ys[-1]))) + 1)
    dense_x = scipy.interpolate.PchipInterpolator(ys, xs)(dense_y)
    dense_x = np.clip(np.rint(dense_x), 0, width - 1).astype(int)
    label = np.zeros((height, width), dtype=np.uint8)
    label[dense_y, dense_x] = 1
    return label
```

- [ ] **Step 6: Run annotation unit tests and one checkpoint smoke test**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_annotation -v`  
Expected: all Task 2 tests PASS.

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -c "from pathlib import Path; from low_light_dataset.annotation import HoseAnnotator; a=HoseAnnotator(Path(r'logs\0209_1509_lr_1e-04_b_64\best_model.pth')); print(a.device)"`  
Expected: prints `cuda:0` or `cuda` and exits 0 without missing/unexpected non-auxiliary key errors.

- [ ] **Step 7: Record checkpoint**

Run: `Get-FileHash low_light_dataset\annotation.py; Get-FileHash tests\test_low_light_annotation.py`  
Expected: SHA-256 values recorded; no Git commit.

---

### Task 3: Artifact Generation and Review Gate

**Files:**
- Create: `low_light_dataset/artifacts.py`
- Create: `scripts/build_low_light_training_dataset.py`
- Test: `tests/test_low_light_artifacts.py`

**Interfaces:**
- Consumes: Task 1 enhancement results and Task 2 predictions/labels.
- Produces: `prepare_dataset(video_path, output_root, checkpoint_path, device) -> Path`.
- Produces: `finalize_labels(output_root, review_path: Path) -> ValidationReport`; the review file must explicitly cover all 60 stems.
- Produces: `validate_prepared_bundle(output_root, require_review=True) -> ValidationReport`.
- CLI phases: `prepare`, `finalize`, `validate`, `merge`.

- [ ] **Step 1: Write failing manifest, pairing, and review-gate tests**

```python
class ReviewGateTests(unittest.TestCase):
    def test_unreviewed_frame_blocks_bundle(self):
        root = make_minimal_bundle(self.tempdir, reviewed=False)
        report = validate_prepared_bundle(root, require_review=True)
        self.assertFalse(report.ok)
        self.assertIn("unreviewed", report.errors[0])

    def test_same_stem_binary_pair_passes(self):
        root = make_minimal_bundle(self.tempdir, reviewed=True)
        report = validate_prepared_bundle(root, require_review=True)
        self.assertTrue(report.ok, report.errors)
```

- [ ] **Step 2: Run tests and verify missing artifacts module failure**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_artifacts -v`  
Expected: FAIL importing `low_light_dataset.artifacts`.

- [ ] **Step 3: Implement process-directory creation and atomic preparation**

```python
ARTIFACT_DIRS = ("original", "enhanced", "label_initial", "label", "overlay")

def create_staging_root(output_root):
    staging = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite existing dataset artifacts")
    for name in ARTIFACT_DIRS:
        (staging / name).mkdir(parents=True, exist_ok=True)
    return staging

def publish_staging(staging, output_root):
    if sum(1 for _ in (staging / "original").glob("*.jpg")) != 60:
        raise RuntimeError("staging dataset does not contain 60 original frames")
    staging.replace(output_root)
```

- [ ] **Step 4: Implement 60-frame prepare loop and manifest fields**

```python
def prepare_dataset(video_path, output_root, checkpoint_path, device="cuda"):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    if capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS) < 300:
        raise ValueError("video is shorter than five minutes")
    annotator = HoseAnnotator(checkpoint_path, device=device)
    staging = create_staging_root(output_root)
    records = []
    for second in sample_times():
        raw, actual = read_video_frame(capture, second)
        enhanced = enhance_low_light(raw)
        stem = f"lowlight_camera_full_rgb_t{second:06d}"
        save_jpeg(staging / "original" / f"{stem}.jpg", raw)
        save_jpeg(staging / "enhanced" / f"{stem}.jpg", enhanced.image)
        original_points = annotator.predict(raw)
        enhanced_points = annotator.predict(enhanced.image)
        fused, warnings = fuse_predictions(original_points, enhanced_points)
        final_points, regularization_warnings = regularize_anchors(fused)
        label = rasterize_centerline(final_points)
        save_label(staging / "label_initial" / f"{stem}.png", label)
        save_overlay(staging / "overlay" / f"{stem}.jpg", enhanced.image, label)
        records.append(build_record(stem, second, actual, enhanced,
                                    original_points, enhanced_points, final_points,
                                    warnings + regularization_warnings))
    capture.release()
    write_manifest(staging / "manifest.csv", records)
    write_annotation_json(staging / "annotation.json", records, reviewed=False)
    write_contact_sheets(staging, records)
    publish_staging(staging, output_root)
    return output_root
```

- [ ] **Step 5: Implement final-label regeneration from reviewed annotation JSON**

```python
def finalize_labels(output_root, review_path):
    document = json.loads((output_root / "annotation.json").read_text(encoding="utf-8"))
    decisions = load_review_decisions(review_path)
    expected = {frame["stem"] for frame in document["frames"]}
    if set(decisions) != expected:
        raise ValueError("review decisions must cover exactly all 60 stems")
    for frame in document["frames"]:
        decision = decisions[frame["stem"]]
        if not decision["approved"]:
            raise ValueError(f"frame was not approved: {frame['stem']}")
        if "anchors" in decision:
            frame["final_anchors"] = decision["anchors"]
            frame["corrected"] = True
        frame["reviewed"] = True
        anchors = [AnchorPrediction(**item) for item in frame["final_anchors"]]
        label = rasterize_centerline(anchors)
        save_label(output_root / "label" / f"{frame['stem']}.png", label)
        enhanced = load_bgr(output_root / "enhanced" / f"{frame['stem']}.jpg")
        save_overlay(output_root / "overlay" / f"{frame['stem']}.jpg", enhanced, label)
    write_json_atomic(output_root / "annotation.json", document)
    return validate_prepared_bundle(output_root, require_review=True)
```

- [ ] **Step 6: Implement CLI phases without importing any training entry point**

```python
def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--review", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--dataset", type=Path, required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--dataset", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-merge")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--baseline", type=Path, required=True)
    return parser
```

- [ ] **Step 7: Run artifact tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_artifacts -v`  
Expected: all Task 3 tests PASS.

- [ ] **Step 8: Record checkpoint**

Run: `Get-FileHash low_light_dataset\artifacts.py; Get-FileHash scripts\build_low_light_training_dataset.py`  
Expected: SHA-256 values recorded; no Git commit.

---

### Task 4: Transactional Dataset Merge

**Files:**
- Create: `low_light_dataset/dataset_merge.py`
- Test: `tests/test_low_light_dataset_merge.py`

**Interfaces:**
- Consumes: reviewed output bundle from Task 3.
- Produces: `snapshot_dataset(dataset_root: Path) -> DatasetSnapshot`.
- Produces: `preflight_merge(bundle_root: Path, dataset_root: Path) -> MergePlan`.
- Produces: `merge_bundle(bundle_root: Path, dataset_root: Path) -> Path` returning receipt path.

- [ ] **Step 1: Write failing collision, leakage, and successful-merge tests**

```python
class MergeTests(unittest.TestCase):
    def test_same_name_different_hash_blocks_merge(self):
        bundle, dataset = make_bundle_and_dataset(self.tempdir, collision=True)
        with self.assertRaisesRegex(ValueError, "name collision"):
            preflight_merge(bundle, dataset)

    def test_test_image_hash_blocks_leakage(self):
        bundle, dataset = make_bundle_and_dataset(self.tempdir, leak=True)
        with self.assertRaisesRegex(ValueError, "test-set leakage"):
            preflight_merge(bundle, dataset)

    def test_successful_merge_preserves_pairs(self):
        bundle, dataset = make_bundle_and_dataset(self.tempdir)
        receipt = merge_bundle(bundle, dataset)
        self.assertTrue(receipt.exists())
        self.assertEqual(stems(dataset / "train" / "pic", ".jpg"),
                         stems(dataset / "train" / "label", ".png"))
```

- [ ] **Step 2: Run tests and verify missing merge module failure**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_dataset_merge -v`  
Expected: FAIL importing `low_light_dataset.dataset_merge`.

- [ ] **Step 3: Implement SHA-256 snapshots and preflight invariants**

```python
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def preflight_merge(bundle_root, dataset_root):
    validation = validate_prepared_bundle(bundle_root, require_review=True)
    if not validation.ok:
        raise ValueError("bundle validation failed: " + "; ".join(validation.errors))
    train = snapshot_split(dataset_root / "train")
    test = snapshot_split(dataset_root / "test")
    if len(train.images) != 4858 or len(train.labels) != 4858:
        raise ValueError("training baseline is not 4858/4858")
    if len(test.images) != 867 or len(test.labels) != 867:
        raise ValueError("test baseline is not 867/867")
    assert_no_name_collisions(bundle_root, train, test)
    assert_no_hash_leakage(bundle_root, test)
    return MergePlan(bundle_root, dataset_root, train, test)
```

- [ ] **Step 4: Implement temporary-copy, hash verification, atomic rename, and receipt**

```python
def merge_bundle(bundle_root, dataset_root):
    plan = preflight_merge(bundle_root, dataset_root)
    created = []
    published = []
    try:
        for source_dir, target_dir, suffix in ((bundle_root / "enhanced", dataset_root / "train" / "pic", ".jpg"),
                                                (bundle_root / "label", dataset_root / "train" / "label", ".png")):
            for source in sorted(source_dir.glob(f"*{suffix}")):
                temporary = target_dir / f".{source.name}.lowlight-staging"
                shutil.copyfile(source, temporary)
                if sha256_file(source) != sha256_file(temporary):
                    raise IOError(f"hash mismatch after copy: {source.name}")
                created.append(temporary)
        for temporary in created:
            final = temporary.parent / temporary.name[1:-len(".lowlight-staging")]
            temporary.replace(final)
            published.append(final)
        post = snapshot_dataset(dataset_root)
        if len(post.train.images) != 4918 or len(post.train.labels) != 4918:
            raise RuntimeError("post-merge training count is not 4918/4918")
        if len(post.test.images) != 867 or len(post.test.labels) != 867:
            raise RuntimeError("test split changed during merge")
        return write_merge_receipt(bundle_root, plan, post)
    except Exception:
        for path in created:
            if path.exists() and path.name.endswith(".lowlight-staging"):
                path.unlink()
        for path in published:
            if path.exists() and path.stem.startswith("lowlight_camera_full_rgb_"):
                path.unlink()
        raise
```

- [ ] **Step 5: Run merge tests and full unit suite**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_low_light_dataset_merge -v`  
Expected: all Task 4 tests PASS.

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest discover -s tests -p "test_low_light_*.py" -v`  
Expected: all low-light tests PASS.

- [ ] **Step 6: Record checkpoint**

Run: `Get-FileHash low_light_dataset\dataset_merge.py; Get-FileHash tests\test_low_light_dataset_merge.py`  
Expected: SHA-256 values recorded; no Git commit.

---

### Task 5: Generate, Review, Correct, Merge, and Verify the Real Dataset

**Files:**
- Generate: `datasets/low_light_camera_full_rgb/**`
- Generate: `datasets/low_light_camera_full_rgb/review_decisions.json`
- Modify by controlled merge: `datasets/newdata/train/pic/*.jpg`
- Modify by controlled merge: `datasets/newdata/train/label/*.png`

**Interfaces:**
- Consumes: CLI and package from Tasks 1–4.
- Produces: 60 reviewed pairs in the process bundle, 60 merged pairs in formal training data, and `merge_receipt.json`.

- [ ] **Step 1: Run the full unit suite before touching real data**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest discover -s tests -p "test_low_light_*.py" -v`  
Expected: all tests PASS.

- [ ] **Step 2: Record immutable baseline counts and hashes**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\build_low_light_training_dataset.py snapshot --dataset datasets\newdata --output datasets\newdata_before_lowlight.json`  
Expected: receipt reports train `4858/4858`, test `867/867`, and matching image/label stems.

- [ ] **Step 3: Prepare the 60-frame artifact bundle**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\build_low_light_training_dataset.py prepare --video "E:\New_DOTEK\数据\camera_full_rgb.mp4" --output datasets\low_light_camera_full_rgb --checkpoint logs\0209_1509_lr_1e-04_b_64\best_model.pth --device cuda`  
Expected: exits 0; creates 60 original JPGs, 60 enhanced JPGs, 60 initial PNGs, 60 overlays, `manifest.csv`, `annotation.json`, and contact sheets.

- [ ] **Step 4: Inspect every enhancement and label overlay**

Open each generated `preview_*.jpg` and `overlay_preview_*.jpg` contact sheet with image inspection. For every frame, verify the enhanced image preserves hose appearance, the red overlay follows the hose center, and no line is projected through invisible regions. Record an explicit decision for all 60 stems in `review_decisions.json`.

Expected: every frame is explicitly inspected; questionable frames are listed with corrected anchor coordinates rather than silently accepted.

- [ ] **Step 5: Apply necessary corrections and finalize labels**

Create `review_decisions.json` with all 60 stems. Approved frames use `{"approved": true}`; corrections additionally contain exact reviewed anchors:

```json
{
  "schema_version": 1,
  "decisions": {
    "lowlight_camera_full_rgb_t000000": {
      "approved": true,
      "anchors": [
        {"y": 220, "x": 318.0, "confidence": 1.0, "source": "reviewed"},
        {"y": 260, "x": 325.0, "confidence": 1.0, "source": "reviewed"},
        {"y": 320, "x": 338.0, "confidence": 1.0, "source": "reviewed"}
      ]
    }
  }
}
```

The actual file must contain all 60 stems. Anchor arrays appear only for frames that need correction, and their coordinates must be read from the real overlays; the example values above document the schema and are not real annotations.

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\build_low_light_training_dataset.py finalize --output datasets\low_light_camera_full_rgb --review datasets\low_light_camera_full_rgb\review_decisions.json`  
Expected: 60 final PNG labels, regenerated overlays, and validation report `ok=true`.

- [ ] **Step 6: Validate the completed bundle independently**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\build_low_light_training_dataset.py validate --output datasets\low_light_camera_full_rgb`  
Expected: reports 60/60 same-stem pairs, all labels binary/nonempty/640×480, all frames reviewed, and no errors.

- [ ] **Step 7: Merge the reviewed bundle into the formal training set**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\build_low_light_training_dataset.py merge --output datasets\low_light_camera_full_rgb --dataset datasets\newdata`  
Expected: merge preflight passes, 120 files are published, and `merge_receipt.json` reports train `4918/4918`, test `867/867`.

- [ ] **Step 8: Verify formal dataset compatibility without training**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -c "from utils_alian.dataset_alian import ClsDataset; from data.constant import my_row_anchor; d=ClsDataset(r'datasets\newdata\train', row_anchor=my_row_anchor, griding_num=50, num_lanes=1); print(len(d), len(d.img_paths), len(d.label_paths)); assert len(d)==4918 and len(d.img_paths)==4918"`  
Expected: prints `4918 4918 4918`.

- [ ] **Step 9: Verify no training side effects**

Run: `Get-Process python,pythonw -ErrorAction SilentlyContinue | Select-Object Id,Path,StartTime`  
Expected: no process command line was started by this workflow for `train_alian.py`; no new model checkpoints or training log directories are created.

- [ ] **Step 10: Run final integrity report**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\build_low_light_training_dataset.py verify-merge --output datasets\low_light_camera_full_rgb --dataset datasets\newdata --baseline datasets\newdata_before_lowlight.json`  
Expected: confirms 60 new reviewed pairs, unchanged 4858 original training pairs, unchanged 867 test pairs, no name/hash leakage, and matching receipt hashes.
