import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from low_light_dataset.candidates import (
    difference_hash,
    extract_candidate_set,
    hamming_distance,
)


def make_video(path: Path, seconds: int = 4, fps: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480)
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer failed")
    for index in range(seconds * fps):
        frame = np.full((480, 640, 3), 35 + index, dtype=np.uint8)
        x = min(600, index * 18)
        cv2.rectangle(frame, (x, 280), (min(639, x + 30), 470), (30, 30, 220), -1)
        writer.write(frame)
    writer.release()


class HashTests(unittest.TestCase):
    def test_difference_hash_and_hamming_distance(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        gradient = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        gradient = np.repeat(gradient[:, :, None], 3, axis=2)
        self.assertEqual(hamming_distance(difference_hash(black), difference_hash(black)), 0)
        self.assertGreater(hamming_distance(difference_hash(black), difference_hash(gradient)), 0)


class CandidateExtractionTests(unittest.TestCase):
    def test_extracts_base_and_bounded_event_frames_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            output = root / "candidates"
            make_video(video)

            summary_path = extract_candidate_set(
                video, output, duration_seconds=4.0,
                analysis_interval_seconds=0.25,
            )

            document = json.loads(summary_path.read_text(encoding="utf-8"))
            with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(document["base_count"], 4)
            self.assertGreaterEqual(document["candidate_count"], 4)
            self.assertLessEqual(document["candidate_count"], 8)
            self.assertEqual(len(rows), document["candidate_count"])
            self.assertEqual(
                [float(row["target_timestamp_seconds"]) for row in rows if row["source"] == "base"],
                [0.0, 1.0, 2.0, 3.0],
            )
            self.assertFalse(list(output.rglob("*.png")))
            for row in rows:
                path = output / "frames" / f"{row['stem']}.jpg"
                with Image.open(path) as image:
                    self.assertEqual(image.size, (640, 480))
            self.assertTrue(list((output / "contact_sheets").glob("*.jpg")))

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            output = root / "candidates"
            make_video(video, seconds=1)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                extract_candidate_set(
                    video, output, duration_seconds=1.0,
                    analysis_interval_seconds=0.25,
                )


if __name__ == "__main__":
    unittest.main()
