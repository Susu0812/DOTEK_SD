import csv
import hashlib
import json
import socket
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np
from werkzeug.test import Client
from werkzeug.wrappers import Response

from low_light_dataset.image_ops import (
    EnhancedFrame,
    EnhancementParams,
    measure_frame,
    save_jpeg,
)
from low_light_dataset.review_service import create_app, find_available_port
from low_light_dataset.review_store import ReviewStore


FINAL_ANCHORS = [
    {"y": 100, "x": 120.0, "confidence": 1.0, "source": "human"},
    {"y": 220, "x": 240.0, "confidence": 1.0, "source": "human"},
    {"y": 340, "x": 360.0, "confidence": 1.0, "source": "human"},
]


class ReviewServiceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.candidate_root = self.root / "candidates"
        self.frames = self.candidate_root / "frames"
        self.frames.mkdir(parents=True)
        self.work_root = self.root / "work"
        self.static_root = self.root / "static"
        self.static_root.mkdir()
        (self.static_root / "app.js").write_bytes(b"console.log('review');")
        (self.static_root / "nested").mkdir()
        (self.static_root / "nested" / "secret.js").write_bytes(b"secret")

        # Manifest order is intentionally not chronological.
        rows = [("frame_c", 30.0, 30), ("frame_a", 10.0, 10), ("frame_b", 20.0, 20)]
        self.manifest_rows = []
        for stem, timestamp, value in rows:
            frame = np.full((480, 640, 3), value, dtype=np.uint8)
            path = self.frames / f"{stem}.jpg"
            save_jpeg(path, frame)
            self.manifest_rows.append(
                {
                    "stem": stem,
                    "target_timestamp_seconds": str(timestamp),
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": "fixture",
                }
            )
        with (self.candidate_root / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=self.manifest_rows[0])
            writer.writeheader()
            writer.writerows(self.manifest_rows)

        self.preannotation_path = self.root / "preannotation.json"
        self.preannotation_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_sha256": "a" * 64,
                    "records": [
                        {
                            "stem": "frame_b",
                            "image_sha256": next(
                                row["image_sha256"]
                                for row in self.manifest_rows
                                if row["stem"] == "frame_b"
                            ),
                            "status": "positive",
                            "notes": "model must not overwrite human notes",
                            "anchors": FINAL_ANCHORS,
                            "warnings": ["model_warning"],
                            "source_metrics": {"final_count": 3},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.app = create_app(
            self.candidate_root,
            self.work_root,
            preannotation_path=self.preannotation_path,
            static_root=self.static_root,
        )
        self.client = Client(self.app, Response)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def candidate_snapshot(self):
        return {
            path.relative_to(self.candidate_root).as_posix(): (
                path.is_file(), path.read_bytes() if path.is_file() else b""
            )
            for path in sorted(self.candidate_root.rglob("*"))
        }

    def patch(self, stem, payload):
        return self.client.patch(
            f"/api/records/{stem}",
            data=json.dumps(payload),
            content_type="application/json",
        )

    @staticmethod
    def fake_enhance(frame):
        image = np.clip(frame.astype(np.int16) + 7, 0, 255).astype(np.uint8)
        return EnhancedFrame(
            image=image,
            before=measure_frame(frame),
            after=measure_frame(image),
            params=EnhancementParams(0.8, 0.2, 1.8, 30.0, 0.25),
        )


class RouteTests(ReviewServiceFixture):
    def test_health_summary_records_detail_neighbors_queue_and_suggestions(self):
        self.assertEqual(
            self.client.get("/health").get_json(),
            {"ok": True, "candidate_count": 3, "record_count": 3},
        )
        summary = self.client.get("/api/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.get_json(), self.app.review_store.summary())

        records_response = self.client.get("/api/records")
        self.assertEqual(records_response.status_code, 200)
        records = records_response.get_json()["records"]
        self.assertEqual([item["stem"] for item in records], ["frame_a", "frame_b", "frame_c"])
        suggested = records[1]
        self.assertEqual(suggested["status"], "unreviewed")
        self.assertEqual(suggested["notes"], "")
        self.assertEqual(suggested["preannotation"]["status"], "positive")
        self.assertEqual(suggested["preannotation"]["source_metrics"], {"final_count": 3})

        detail = self.client.get("/api/records/frame_b").get_json()
        self.assertEqual(detail["previous_stem"], "frame_a")
        self.assertEqual(detail["next_stem"], "frame_c")
        self.assertIsNone(self.client.get("/api/records/frame_a").get_json()["previous_stem"])
        self.assertIsNone(self.client.get("/api/records/frame_c").get_json()["next_stem"])

        queue = self.client.get("/api/review-queue").get_json()
        self.assertEqual(queue["stems"], ["frame_a", "frame_b", "frame_c"])
        self.assertEqual([item["stem"] for item in queue["records"]], queue["stems"])

    def test_original_media_static_file_and_all_method_contracts(self):
        expected = (self.frames / "frame_a.jpg").read_bytes()
        media = self.client.get("/media/original/frame_a.jpg")
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.data, expected)
        self.assertEqual(media.mimetype, "image/jpeg")
        static = self.client.get("/static/app.js")
        self.assertEqual(static.status_code, 200)
        self.assertEqual(static.data, b"console.log('review');")

        for method, path in (("post", "/health"), ("post", "/api/records"), ("get", "/api/records/frame_a")):
            with self.subTest(method=method, path=path):
                response = getattr(self.client, method)(path)
                self.assertIn(response.status_code, (200, 405))
        self.assertEqual(self.client.post("/health").status_code, 405)
        self.assertEqual(self.client.post("/api/records").status_code, 405)

    def test_traversal_encoded_traversal_unknown_stems_and_suffixes_are_rejected(self):
        paths = (
            "/api/records/unknown",
            "/api/records/..%2Fframe_a",
            "/media/original/unknown.jpg",
            "/media/original/../frame_a.jpg",
            "/media/original/%2e%2e%2fframe_a.jpg",
            "/media/original/%2e%2e%5coutside.jpg",
            "/media/original/frame_a.png",
            "/media/original/frame_a.jpg.exe",
            "/media/enhanced/unknown.jpg",
            "/static/nested/secret.js",
            "/static/..%2Fnested%2Fsecret.js",
            "/static/%2e%2e%5cnested%5csecret.js",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_head_and_options_are_not_implicitly_added_to_get_routes(self):
        for method in ("head", "options"):
            with self.subTest(method=method):
                response = getattr(self.client, method)("/health")
                self.assertEqual(response.status_code, 405)
                if method == "head":
                    self.assertEqual(response.data, b"")
                else:
                    self.assertEqual(response.get_json()["error"], "method_not_allowed")

    def test_original_media_symlink_escape_is_rejected_before_hash_or_read(self):
        original = self.frames / "frame_a.jpg"
        outside = self.root / "outside-original.jpg"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        try:
            original.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"file symlink creation unavailable: {exc}")

        outside_before = outside.read_bytes()
        with mock.patch.object(
            self.app.review_store,
            "_sha256",
            wraps=self.app.review_store._sha256,
        ) as candidate_hash:
            response = self.client.get("/media/original/frame_a.jpg")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json(), {"error": "candidate_changed"})
        candidate_hash.assert_not_called()
        self.assertEqual(outside.read_bytes(), outside_before)


class InitializationSecurityTests(unittest.TestCase):
    def test_unsafe_manifest_stems_reject_before_candidate_access_or_state_write(self):
        malicious_stems = (
            "",
            ".",
            "..",
            "../outside",
            "..\\outside",
            "..\\..\\outside",
            "nested/name",
            "nested\\name",
            "/absolute",
            "C:\\absolute",
            "name.jpg",
            "with space",
            "control\x01name",
            "nul\x00name",
            "非ascii",
        )
        for stem in malicious_stems:
            with self.subTest(stem=repr(stem)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate_root = root / "candidates"
                frames = candidate_root / "frames"
                frames.mkdir(parents=True)
                sentinel = root / "outside.jpg"
                sentinel.write_bytes(b"outside-sentinel")
                (candidate_root / "outside.jpg").write_bytes(b"outside-sentinel")
                (candidate_root / "manifest.csv").write_text(
                    "stem,image_sha256\n"
                    + stem
                    + ","
                    + hashlib.sha256(b"outside-sentinel").hexdigest()
                    + "\n",
                    encoding="utf-8",
                )

                work_root = root / "work"
                sentinel_before = sentinel.read_bytes()
                caught = None
                with mock.patch.object(
                    ReviewStore,
                    "_sha256",
                    side_effect=AssertionError("unsafe candidate access attempted"),
                ) as candidate_hash:
                    try:
                        create_app(candidate_root, work_root)
                    except BaseException as exc:
                        caught = exc

                self.assertIsInstance(caught, ValueError)
                self.assertIn("unsafe_manifest_stem", str(caught))
                candidate_hash.assert_not_called()
                self.assertFalse(work_root.exists())
                self.assertEqual(sentinel.read_bytes(), sentinel_before)


class PatchTests(ReviewServiceFixture):
    def test_draft_then_finalize_sets_utc_first_timestamp_and_preserves_it(self):
        draft = self.patch("frame_a", {"revision": 0, "notes": "draft"})
        self.assertEqual(draft.status_code, 200)
        self.assertEqual(draft.get_json()["revision"], 1)
        self.assertIsNone(draft.get_json()["first_reviewed_at"])

        finalized = self.patch(
            "frame_a",
            {"revision": 1, "action": "finalize", "status": "positive", "anchors": FINAL_ANCHORS},
        )
        self.assertEqual(finalized.status_code, 200)
        first = finalized.get_json()["first_reviewed_at"]
        parsed = datetime.fromisoformat(first)
        self.assertEqual(parsed.utcoffset(), timedelta(0))
        self.assertLess(abs(datetime.now(timezone.utc) - parsed), timedelta(seconds=5))

        updated = self.patch("frame_a", {"revision": 2, "notes": "after finalize"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.get_json()["first_reviewed_at"], first)

    def test_final_status_in_draft_is_treated_as_finalize(self):
        response = self.patch(
            "frame_a", {"revision": 0, "status": "positive", "anchors": FINAL_ANCHORS}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.get_json()["first_reviewed_at"])

    def test_second_review_sets_server_timestamps_and_preserves_existing_first(self):
        first = self.patch(
            "frame_a", {"revision": 0, "status": "positive", "anchors": FINAL_ANCHORS}
        ).get_json()["first_reviewed_at"]
        second = self.patch(
            "frame_a", {"revision": 1, "action": "second_review", "status": "positive"}
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.get_json()["first_reviewed_at"], first)
        self.assertIsNotNone(second.get_json()["second_reviewed_at"])
        self.assertTrue(second.get_json()["second_review_required"])

        simultaneous = self.patch(
            "frame_b",
            {"revision": 0, "action": "second_review", "status": "positive", "anchors": FINAL_ANCHORS},
        )
        self.assertEqual(simultaneous.status_code, 200)
        self.assertIsNotNone(simultaneous.get_json()["first_reviewed_at"])
        self.assertIsNotNone(simultaneous.get_json()["second_reviewed_at"])

    def test_client_timestamps_malformed_controls_and_invalid_final_states_are_400(self):
        cases = (
            ({"revision": 0, "first_reviewed_at": None}, "client_timestamp_forbidden"),
            ({"revision": 0, "second_reviewed_at": "x"}, "client_timestamp_forbidden"),
            ({"revision": True, "notes": "x"}, "invalid_revision"),
            ({"revision": "0", "notes": "x"}, "invalid_revision"),
            ({"revision": 0, "action": "approve"}, "invalid_action"),
            ({"revision": 0, "action": []}, "invalid_action"),
            ({"revision": 0, "action": "finalize", "status": "unreviewed"}, "final_status_required"),
            ({"revision": 0, "action": "second_review", "status": "needs_second_review"}, "final_status_required"),
            ({"revision": 0, "status": {}}, "validation_failed"),
            ({"revision": 0, "origin": "system", "status": "positive", "anchors": FINAL_ANCHORS}, "invalid_patch"),
            ({"revision": 0, "second_review_required": True}, "invalid_patch"),
        )
        for payload, error in cases:
            with self.subTest(payload=payload):
                response = self.patch("frame_a", payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"], error)
                self.assertEqual(self.app.review_store.get("frame_a").revision, 0)

        malformed = self.client.patch(
            "/api/records/frame_a", data="{", content_type="application/json"
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.get_json()["error"], "invalid_json")

    def test_revision_conflict_and_record_validation_do_not_mutate(self):
        state = (self.work_root / "annotation_state.json").read_bytes()
        stale = self.patch("frame_a", {"revision": 9, "notes": "stale"})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.get_json()["error"], "revision_conflict")
        self.assertEqual((self.work_root / "annotation_state.json").read_bytes(), state)

        invalid = self.patch("frame_a", {"revision": 0, "status": "positive", "anchors": []})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.get_json()["error"], "validation_failed")
        self.assertEqual((self.work_root / "annotation_state.json").read_bytes(), state)

    def test_candidate_change_is_409_and_save_or_audit_failures_are_stable_500(self):
        original = self.frames / "frame_a.jpg"
        original.write_bytes(original.read_bytes() + b"changed")
        changed = self.patch("frame_a", {"revision": 0, "notes": "no"})
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(changed.get_json(), {"error": "candidate_changed"})

        # Restore exact canonical bytes and exercise both persistence failure paths.
        original.write_bytes(original.read_bytes()[:-7])
        with mock.patch.object(self.app.review_store, "_write_state", side_effect=OSError("private path")):
            failed = self.patch("frame_a", {"revision": 0, "notes": "no"})
        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.get_json(), {"error": "save_failed"})
        self.assertNotIn(str(self.root), failed.get_data(as_text=True))

        with mock.patch.object(self.app.review_store, "_append_history", side_effect=OSError("private path")):
            audit = self.patch("frame_b", {"revision": 0, "notes": "saved before audit"})
        self.assertEqual(audit.status_code, 500)
        self.assertEqual(audit.get_json(), {"error": "save_failed"})
        self.assertNotIn(str(self.root), audit.get_data(as_text=True))

    def test_client_actor_cannot_choose_a_nonhuman_origin(self):
        response = self.patch(
            "frame_a",
            {
                "revision": 0,
                "actor": "model",
                "status": "positive",
                "anchors": FINAL_ANCHORS,
            },
        )
        self.assertEqual(response.status_code, 200)
        event = json.loads(
            (self.work_root / "annotation_history.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(event["actor"], "local_web_reviewer")
        self.assertEqual(event["origin"], "human")


class EnhancedCacheTests(ReviewServiceFixture):
    def test_enhanced_cache_generation_reuse_and_candidate_tree_read_only(self):
        before = self.candidate_snapshot()
        with mock.patch(
            "low_light_dataset.review_service.enhance_low_light",
            side_effect=self.fake_enhance,
        ) as enhance:
            first = self.client.get("/media/enhanced/frame_a.jpg")
            second = self.client.get("/media/enhanced/frame_a.jpg")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.data, first.data)
        self.assertEqual(enhance.call_count, 1)
        self.assertEqual(self.candidate_snapshot(), before)

        cache = self.work_root / "enhanced_preview_cache"
        sidecar = json.loads((cache / "frame_a.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["schema_version"], 1)
        self.assertEqual(sidecar["source_sha256"], self.manifest_rows[1]["image_sha256"])
        self.assertEqual(sidecar["output"]["width"], 640)
        self.assertEqual(sidecar["output"]["height"], 480)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(first.data).hexdigest())
        self.assertFalse(list(cache.glob(".*.tmp")))

    def test_corrupt_or_stale_cache_and_sidecar_regenerate(self):
        cache = self.work_root / "enhanced_preview_cache"
        scenarios = ("corrupt_jpeg", "bad_json", "stale_hash", "bad_dimensions")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                with mock.patch(
                    "low_light_dataset.review_service.enhance_low_light",
                    side_effect=self.fake_enhance,
                ) as enhance:
                    self.assertEqual(self.client.get("/media/enhanced/frame_b.jpg").status_code, 200)
                    jpeg = cache / "frame_b.jpg"
                    sidecar_path = cache / "frame_b.json"
                    if scenario == "corrupt_jpeg":
                        jpeg.write_bytes(b"not a jpeg")
                    elif scenario == "bad_json":
                        sidecar_path.write_text("{", encoding="utf-8")
                    elif scenario == "stale_hash":
                        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
                        metadata["source_sha256"] = "0" * 64
                        sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
                    else:
                        small = np.zeros((240, 320, 3), dtype=np.uint8)
                        save_jpeg(jpeg, small)
                        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
                        metadata["output"]["width"] = 320
                        metadata["output"]["height"] = 240
                        metadata["output"]["sha256"] = hashlib.sha256(jpeg.read_bytes()).hexdigest()
                        sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
                    self.assertEqual(self.client.get("/media/enhanced/frame_b.jpg").status_code, 200)
                self.assertEqual(enhance.call_count, 2)
                for path in (cache / "frame_b.jpg", cache / "frame_b.json"):
                    path.unlink()

    def test_cache_root_symlink_escape_is_rejected_without_outside_writes(self):
        outside = self.root / "outside-cache"
        outside.mkdir()
        cache_root = self.work_root / "enhanced_preview_cache"
        try:
            cache_root.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlink creation unavailable: {exc}")

        before = {
            path.relative_to(outside).as_posix(): path.read_bytes()
            for path in outside.rglob("*")
            if path.is_file()
        }
        with mock.patch(
            "low_light_dataset.review_service.enhance_low_light",
            side_effect=AssertionError("enhancement must not run through cache link"),
        ) as enhance:
            response = self.client.get("/media/enhanced/frame_a.jpg")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "save_failed"})
        enhance.assert_not_called()
        after = {
            path.relative_to(outside).as_posix(): path.read_bytes()
            for path in outside.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_cache_file_symlink_escape_is_rejected_before_outside_read(self):
        with mock.patch(
            "low_light_dataset.review_service.enhance_low_light",
            side_effect=self.fake_enhance,
        ):
            generated = self.client.get("/media/enhanced/frame_a.jpg")
        self.assertEqual(generated.status_code, 200)

        cache_path = self.work_root / "enhanced_preview_cache" / "frame_a.jpg"
        outside = self.root / "outside-preview.jpg"
        outside.write_bytes(cache_path.read_bytes())
        cache_path.unlink()
        try:
            cache_path.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"file symlink creation unavailable: {exc}")

        outside_before = outside.read_bytes()
        with mock.patch(
            "low_light_dataset.review_service.enhance_low_light",
            side_effect=AssertionError("enhancement must not follow cache link"),
        ) as enhance:
            response = self.client.get("/media/enhanced/frame_a.jpg")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "save_failed"})
        enhance.assert_not_called()
        self.assertEqual(outside.read_bytes(), outside_before)


class PortTests(unittest.TestCase):
    def test_occupied_start_falls_back_and_returned_probe_socket_is_closed(self):
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        start = occupied.getsockname()[1]
        try:
            port = find_available_port(start=start, end=start + 1)
            self.assertEqual(port, start + 1)
            reusable = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                reusable.bind(("127.0.0.1", port))
            finally:
                reusable.close()
        finally:
            occupied.close()

    def test_occupied_8765_returns_8766_when_8766_is_available(self):
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            occupied.bind(("127.0.0.1", 8765))
            occupied.listen(1)
        except OSError:
            occupied.close()
            self.skipTest("port 8765 is already occupied by an external process")
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", 8766))
            except OSError:
                self.skipTest("port 8766 is already occupied by an external process")
            finally:
                probe.close()
            self.assertEqual(find_available_port(), 8766)
        finally:
            occupied.close()

    def test_fully_occupied_range_has_stable_error(self):
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        first.bind(("127.0.0.1", 0))
        first.listen(1)
        port = first.getsockname()[1]
        try:
            with self.assertRaisesRegex(RuntimeError, f"^no_loopback_port_available_{port}_{port}$"):
                find_available_port(start=port, end=port)
        finally:
            first.close()

    def test_nonliteral_hosts_and_invalid_bounds_reject(self):
        for host in ("localhost", "0.0.0.0", "::1", "127.0.0.01", " 127.0.0.1"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                find_available_port(host=host)
        for start, end in ((0, 1), (1, 65536), (2, 1), (True, 2), (1, False)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                find_available_port(start=start, end=end)


if __name__ == "__main__":
    unittest.main()
