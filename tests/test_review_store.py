import base64
import copy
import csv
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from low_light_dataset.review_models import ReviewStatus
from low_light_dataset.review_store import (
    AuditError,
    CandidateChangedError,
    ReviewStateError,
    ReviewStore,
    RevisionConflict,
)


JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQF"
    "BgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEI"
    "I0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNk"
    "ZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLD"
    "xMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEB"
    "AQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZH"
    "SElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaan"
    "qKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9o"
    "ADAMBAAIRAxEAPwDgqKKK8M/VD//Z"
)


class ReviewStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.candidate_root = self.root / "candidates"
        self.frames = self.candidate_root / "frames"
        self.frames.mkdir(parents=True)
        self.work_root = self.root / "work"
        self.manifest = self.candidate_root / "manifest.csv"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_candidate(self, stem, content=JPEG_BYTES):
        path = self.frames / f"{stem}.jpg"
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def write_manifest(self, rows, fieldnames=("stem", "image_sha256")):
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def prepare(self, stems=("sample_b", "sample_a")):
        rows = []
        for index, stem in enumerate(stems):
            digest = self.write_candidate(stem, JPEG_BYTES + bytes([index]))
            rows.append({"stem": stem, "image_sha256": digest})
        self.write_manifest(rows)
        store = ReviewStore(self.work_root, self.candidate_root)
        state = store.initialize(self.manifest)
        return store, state, rows

    def valid_positive_patch(self):
        return {
            "status": "positive",
            "anchors": [
                {"y": 100, "x": 150.0, "confidence": 1.0, "source": "human"},
                {"y": 200, "x": 250.0, "confidence": 1.0, "source": "human"},
                {"y": 300, "x": 350.0, "confidence": 1.0, "source": "human"},
            ],
            "first_reviewed_at": "2026-07-17T00:00:00+00:00",
        }

    def persistence_snapshot(self):
        names = (
            "annotation_state.json",
            "annotation_state.json.last-good",
            "annotation_state.json.tmp",
            "annotation_history.jsonl",
        )
        return {
            name: (self.work_root / name).read_bytes()
            if (self.work_root / name).exists()
            else None
            for name in names
        }

    def read_history(self):
        return [
            json.loads(line)
            for line in (self.work_root / "annotation_history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def write_history(self, events):
        (self.work_root / "annotation_history.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )


class InitializationTests(ReviewStoreTestCase):
    def test_initialize_preserves_manifest_hashes_and_default_records(self):
        _, state, rows = self.prepare()

        self.assertEqual(state["schema_version"], 2)
        self.assertEqual(
            [record["stem"] for record in state["records"]],
            ["sample_a", "sample_b"],
        )
        by_stem = {record["stem"]: record for record in state["records"]}
        for row in rows:
            record = by_stem[row["stem"]]
            self.assertEqual(record["image_sha256"], row["image_sha256"])
            self.assertEqual(record["revision"], 0)
            self.assertEqual(record["status"], "unreviewed")
            self.assertEqual(record["anchors"], [])
            self.assertEqual(record["interference_tags"], [])
            self.assertIsNone(record["exclusion_reason"])
            self.assertEqual(record["warnings"], [])
            self.assertFalse(record["suggestion_modified"])
            self.assertIsNone(record["first_reviewed_at"])
            self.assertFalse(record["second_review_required"])
            self.assertIsNone(record["second_reviewed_at"])
            self.assertEqual(record["notes"], "")

    def test_initialize_is_idempotent_for_identical_valid_state(self):
        store, first, _ = self.prepare()
        state_path = self.work_root / "annotation_state.json"
        original_bytes = state_path.read_bytes()

        second = store.initialize(self.manifest)

        self.assertEqual(second, first)
        self.assertEqual(state_path.read_bytes(), original_bytes)
        self.assertFalse((self.work_root / "annotation_state.json.last-good").exists())

    def test_initialize_refuses_different_existing_manifest(self):
        store, _, _ = self.prepare(("sample_a",))
        digest = self.write_candidate("sample_b", JPEG_BYTES + b"new")
        self.write_manifest([{"stem": "sample_b", "image_sha256": digest}])

        with self.assertRaisesRegex(ReviewStateError, "manifest does not match"):
            store.initialize(self.manifest)

    def test_initialize_rejects_duplicate_and_empty_manifests(self):
        digest = self.write_candidate("sample_a")
        duplicate = [
            {"stem": "sample_a", "image_sha256": digest},
            {"stem": "sample_a", "image_sha256": digest},
        ]
        self.write_manifest(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate stem"):
            ReviewStore(self.work_root, self.candidate_root).initialize(self.manifest)

        self.write_manifest([])
        with self.assertRaisesRegex(ValueError, "empty manifest"):
            ReviewStore(self.work_root, self.candidate_root).initialize(self.manifest)

    def test_initialize_rejects_missing_columns_and_candidate(self):
        self.write_manifest([{"stem": "sample_a"}], fieldnames=("stem",))
        with self.assertRaisesRegex(ValueError, "required columns"):
            ReviewStore(self.work_root, self.candidate_root).initialize(self.manifest)

        self.write_manifest([{"stem": "missing", "image_sha256": "a" * 64}])
        with self.assertRaises(FileNotFoundError):
            ReviewStore(self.work_root, self.candidate_root).initialize(self.manifest)

    def test_initialize_rejects_hash_mismatch(self):
        self.write_candidate("sample_a")
        self.write_manifest([{"stem": "sample_a", "image_sha256": "a" * 64}])

        with self.assertRaisesRegex(CandidateChangedError, "sample_a"):
            ReviewStore(self.work_root, self.candidate_root).initialize(self.manifest)

    def test_state_json_is_deterministic_and_temporary_is_removed(self):
        self.prepare(("z", "a", "m"))
        state_path = self.work_root / "annotation_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual([item["stem"] for item in state["records"]], ["a", "m", "z"])
        self.assertEqual(
            state_path.read_text(encoding="utf-8"),
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        self.assertFalse((self.work_root / "annotation_state.json.tmp").exists())


class UpdateTests(ReviewStoreTestCase):
    def test_operations_before_initialize_raise_file_not_found(self):
        store = ReviewStore(self.work_root, self.candidate_root)
        for operation in (
            lambda: store.summary(),
            lambda: store.get("sample"),
            lambda: store.update("sample", {}, 0),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(FileNotFoundError):
                    operation()

    def test_get_unknown_and_update_stale_revision(self):
        store, _, _ = self.prepare(("sample_a",))

        with self.assertRaises(KeyError):
            store.get("unknown")
        with self.assertRaisesRegex(RevisionConflict, "sample_a"):
            store.update("sample_a", {"notes": "stale"}, 1)
        self.assertEqual(store.get("sample_a").revision, 0)

    def test_update_rejects_forbidden_keys_and_invalid_record_without_writes(self):
        store, _, _ = self.prepare(("sample_a",))
        state_path = self.work_root / "annotation_state.json"
        original_state = state_path.read_bytes()
        history_path = self.work_root / "annotation_history.jsonl"

        for key in ("stem", "image_sha256", "revision", "unexpected"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "editable"):
                    store.update("sample_a", {key: "bad"}, 0)
        with self.assertRaisesRegex(ValueError, "positive_requires_three_anchors"):
            store.update("sample_a", {"status": "positive"}, 0)

        self.assertEqual(state_path.read_bytes(), original_state)
        self.assertFalse(history_path.exists())

    def test_successful_update_increments_once_and_appends_full_history(self):
        store, _, _ = self.prepare(("sample_a",))
        before = store.get("sample_a")

        after = store.update(
            "sample_a", self.valid_positive_patch(), 0, actor="reviewer-1"
        )

        self.assertEqual(after.revision, 1)
        self.assertEqual(after.status, ReviewStatus.POSITIVE)
        self.assertEqual(store.get("sample_a"), after)
        line = (self.work_root / "annotation_history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(line), 1)
        event = json.loads(line[0])
        self.assertEqual(event["actor"], "reviewer-1")
        self.assertEqual(event["origin"], "human")
        self.assertEqual(event["stem"], "sample_a")
        self.assertEqual(event["prior_revision"], 0)
        self.assertEqual(event["new_revision"], 1)
        self.assertEqual(event["before"]["revision"], before.revision)
        self.assertEqual(event["after"]["revision"], after.revision)
        self.assertRegex(event["time"], r"\+00:00$")
        self.assertTrue((self.work_root / "annotation_state.json.last-good").exists())
        self.assertFalse((self.work_root / "annotation_state.json.tmp").exists())

    def test_system_and_model_origins_cannot_finalize_or_set_first_reviewed_at(self):
        stems = tuple(f"attempt_{index}" for index in range(8))
        store, _, _ = self.prepare(stems)
        final_patches = (
            self.valid_positive_patch(),
            {
                "status": "hard_negative",
                "interference_tags": ["cable_or_rope"],
                "first_reviewed_at": "2026-07-17T00:00:00+00:00",
            },
            {
                "status": "excluded",
                "exclusion_reason": "no_training_value",
                "first_reviewed_at": "2026-07-17T00:00:00+00:00",
            },
        )
        attempts = []
        for origin in ("system", "model"):
            attempts.extend((origin, patch) for patch in final_patches)
            attempts.append(
                (
                    origin,
                    {"first_reviewed_at": "2026-07-17T00:00:00+00:00"},
                )
            )

        for stem, (origin, patch) in zip(stems, attempts):
            with self.subTest(origin=origin, patch=patch):
                before = self.persistence_snapshot()
                with self.assertRaises(PermissionError):
                    store.update(
                        stem,
                        patch,
                        0,
                        actor="detector-v3",
                        origin=origin,
                    )
                self.assertEqual(self.persistence_snapshot(), before)

    def test_nonhuman_suggestion_records_origin_without_approving(self):
        store, _, _ = self.prepare(("sample_a",))

        after = store.update(
            "sample_a",
            {"notes": "model suggestion only"},
            0,
            actor="detector-v3",
            origin="model",
        )

        self.assertEqual(after.status, ReviewStatus.UNREVIEWED)
        event = self.read_history()[0]
        self.assertEqual(event["actor"], "detector-v3")
        self.assertEqual(event["origin"], "model")

    def test_nonhuman_cannot_modify_an_existing_final_record(self):
        store, _, _ = self.prepare(("system_record", "model_record"))
        store.update("system_record", self.valid_positive_patch(), 0)
        store.update("model_record", self.valid_positive_patch(), 0)

        for origin in ("system", "model"):
            with self.subTest(origin=origin):
                stem = f"{origin}_record"
                before = self.persistence_snapshot()
                with self.assertRaises(PermissionError):
                    store.update(
                        stem,
                        {"notes": "non-human post-approval edit"},
                        1,
                        actor="detector-v3",
                        origin=origin,
                    )
                self.assertEqual(self.persistence_snapshot(), before)

    def test_nonhuman_can_explicitly_return_final_record_to_unapproved_state(self):
        store, _, _ = self.prepare(("sample_a",))
        store.update("sample_a", self.valid_positive_patch(), 0)

        after = store.update(
            "sample_a",
            {
                "status": "needs_second_review",
                "first_reviewed_at": None,
                "second_review_required": True,
            },
            1,
            actor="detector-v3",
            origin="model",
        )

        self.assertEqual(after.status, ReviewStatus.NEEDS_SECOND_REVIEW)
        self.assertIsNone(after.first_reviewed_at)
        self.assertFalse(store.summary()["audit_incomplete"])

    def test_invalid_actor_or_origin_has_no_persistence_side_effects(self):
        store, _, _ = self.prepare(("bad_actor", "bad_origin", "non_string_origin"))

        for stem, arguments in (
            ("bad_actor", {"actor": " ", "origin": "human"}),
            ("bad_origin", {"actor": "reviewer", "origin": "robot"}),
            ("non_string_origin", {"actor": "reviewer", "origin": []}),
        ):
            with self.subTest(arguments=arguments):
                before = self.persistence_snapshot()
                with self.assertRaises(ValueError):
                    store.update(stem, {"notes": "not written"}, 0, **arguments)
                self.assertEqual(self.persistence_snapshot(), before)

    def test_state_write_failure_does_not_append_history(self):
        store, _, _ = self.prepare(("sample_a",))
        history_path = self.work_root / "annotation_history.jsonl"

        with mock.patch.object(store, "_write_state", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                store.update("sample_a", {"notes": "not written"}, 0)

        self.assertFalse(history_path.exists())
        self.assertEqual(store.get("sample_a").revision, 0)

    def test_atomic_replace_failure_keeps_current_and_history_unchanged(self):
        store, _, _ = self.prepare(("sample_a",))
        state_path = self.work_root / "annotation_state.json"
        original_state = state_path.read_bytes()

        with mock.patch.object(
            Path, "replace", autospec=True, side_effect=OSError("replace failed")
        ):
            with self.assertRaisesRegex(OSError, "replace failed"):
                store.update("sample_a", {"notes": "not published"}, 0)

        self.assertEqual(state_path.read_bytes(), original_state)
        self.assertEqual(
            (self.work_root / "annotation_state.json.last-good").read_bytes(),
            original_state,
        )
        self.assertTrue((self.work_root / "annotation_state.json.tmp").exists())
        self.assertFalse((self.work_root / "annotation_history.jsonl").exists())

    def test_history_failure_keeps_state_and_sets_audit_incomplete(self):
        store, _, _ = self.prepare(("sample_a",))

        with mock.patch.object(
            store, "_append_history", side_effect=OSError("audit disk full")
        ):
            with self.assertRaisesRegex(AuditError, "audit disk full"):
                store.update("sample_a", {"notes": "published"}, 0)

        self.assertEqual(store.get("sample_a").revision, 1)
        self.assertTrue(store.summary()["audit_incomplete"])
        reopened = ReviewStore(self.work_root, self.candidate_root)
        self.assertTrue(reopened.summary()["audit_incomplete"])

    def test_candidate_mutation_is_detected_by_get_and_update(self):
        store, _, _ = self.prepare(("sample_a",))
        (self.frames / "sample_a.jpg").write_bytes(b"changed")

        with self.assertRaisesRegex(CandidateChangedError, "sample_a"):
            store.get("sample_a")
        with self.assertRaisesRegex(CandidateChangedError, "sample_a"):
            store.update("sample_a", {"notes": "no"}, 0)

    def test_concurrent_instances_serialize_revision_check_and_commit(self):
        first, _, _ = self.prepare(("sample_a",))
        second = ReviewStore(self.work_root, self.candidate_root)
        first_reached_write = threading.Event()
        release_first = threading.Event()
        original_write = first._write_state
        results = []
        errors = []

        def paused_write(state):
            first_reached_write.set()
            if not release_first.wait(5):
                raise TimeoutError("test did not release first writer")
            original_write(state)

        def perform(store, note):
            try:
                results.append(store.update("sample_a", {"notes": note}, 0))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(first, "_write_state", side_effect=paused_write):
            first_thread = threading.Thread(target=perform, args=(first, "first"))
            second_thread = threading.Thread(target=perform, args=(second, "second"))
            first_thread.start()
            self.assertTrue(first_reached_write.wait(5))
            second_thread.start()
            second_thread.join(0.5)
            release_first.set()
            first_thread.join(5)
            second_thread.join(5)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RevisionConflict)
        self.assertEqual(ReviewStore(self.work_root, self.candidate_root).get("sample_a"), results[0])
        self.assertEqual(len(self.read_history()), 1)


class RecoveryAndSummaryTests(ReviewStoreTestCase):
    def test_corrupt_current_recovers_from_valid_last_good_without_overwrite(self):
        store, _, _ = self.prepare(("sample_a",))
        store.update("sample_a", {"notes": "revision one"}, 0)
        state_path = self.work_root / "annotation_state.json"
        corrupt = b"{not-json"
        state_path.write_bytes(corrupt)

        summary = store.summary()

        self.assertTrue(summary["recovered_from_last_good"])
        self.assertEqual(store.get("sample_a").revision, 0)
        self.assertEqual(state_path.read_bytes(), corrupt)

    def test_update_rejects_recovered_state_without_touching_persistence_files(self):
        store, _, _ = self.prepare(("sample_a",))
        store.update("sample_a", {"notes": "revision one"}, 0)
        (self.work_root / "annotation_state.json").write_bytes(b"{corrupt-current")
        (self.work_root / "annotation_state.json.tmp").write_bytes(b"stale-temp")
        before = self.persistence_snapshot()

        with self.assertRaisesRegex(ReviewStateError, "recovered"):
            store.update("sample_a", {"notes": "must not publish"}, 0)

        self.assertEqual(self.persistence_snapshot(), before)

    def test_corrupt_current_and_backup_raise_stable_state_error(self):
        store, _, _ = self.prepare(("sample_a",))
        store.update("sample_a", {"notes": "revision one"}, 0)
        (self.work_root / "annotation_state.json").write_text("bad", encoding="utf-8")
        (self.work_root / "annotation_state.json.last-good").write_text(
            "also bad", encoding="utf-8"
        )

        with self.assertRaisesRegex(ReviewStateError, "no valid review state"):
            store.summary()

    def test_summary_reports_stale_temp_and_all_status_counts(self):
        stems = ("unreviewed", "positive", "negative", "excluded", "second")
        store, _, _ = self.prepare(stems)
        store.update("positive", self.valid_positive_patch(), 0)
        store.update(
            "negative",
            {
                "status": "hard_negative",
                "interference_tags": ["cable_or_rope"],
                "first_reviewed_at": "2026-07-17T00:00:00+00:00",
            },
            0,
        )
        store.update(
            "excluded",
            {
                "status": "excluded",
                "exclusion_reason": "no_training_value",
                "first_reviewed_at": "2026-07-17T00:00:00+00:00",
            },
            0,
        )
        store.update(
            "second",
            {"status": "needs_second_review", "second_review_required": True},
            0,
        )
        (self.work_root / "annotation_state.json.tmp").write_text(
            "stale", encoding="utf-8"
        )

        summary = store.summary()

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["total"], 5)
        self.assertEqual(
            summary["counts"],
            {
                "unreviewed": 1,
                "positive": 1,
                "hard_negative": 1,
                "excluded": 1,
                "needs_second_review": 1,
            },
        )
        self.assertFalse(summary["recovered_from_last_good"])
        self.assertTrue(summary["stale_temporary_present"])
        self.assertFalse(summary["audit_incomplete"])


class AuditValidationTests(ReviewStoreTestCase):
    def prepare_history(self, update_count=1):
        store, _, _ = self.prepare(("sample_a",))
        for revision in range(update_count):
            store.update("sample_a", {"notes": f"revision {revision + 1}"}, revision)
        events = self.read_history()
        for event in events:
            event.setdefault("origin", "human")
        self.write_history(events)
        return store, events

    def test_reopen_accepts_a_structurally_valid_contiguous_history(self):
        self.prepare_history(2)

        reopened = ReviewStore(self.work_root, self.candidate_root)

        self.assertFalse(reopened.summary()["audit_incomplete"])

    def test_reopen_flags_malformed_or_missing_required_event_fields(self):
        _, events = self.prepare_history()
        history_path = self.work_root / "annotation_history.jsonl"

        history_path.write_text("{not-json\n", encoding="utf-8")
        self.assertTrue(ReviewStore(self.work_root, self.candidate_root).summary()["audit_incomplete"])

        for field in (
            "time",
            "actor",
            "origin",
            "stem",
            "prior_revision",
            "new_revision",
            "before",
            "after",
        ):
            with self.subTest(field=field):
                event = copy.deepcopy(events[0])
                del event[field]
                self.write_history([event])
                self.assertTrue(
                    ReviewStore(self.work_root, self.candidate_root).summary()[
                        "audit_incomplete"
                    ]
                )

    def test_reopen_flags_invalid_time_actor_or_origin(self):
        _, events = self.prepare_history()
        invalid_values = (
            ("time", "not-a-time"),
            ("time", "2026-07-17T00:00:00"),
            ("time", "2026-07-17T08:00:00+08:00"),
            ("actor", ""),
            ("actor", "   "),
            ("origin", "robot"),
            ("origin", []),
        )

        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                event = copy.deepcopy(events[0])
                event[field] = value
                self.write_history([event])
                self.assertTrue(
                    ReviewStore(self.work_root, self.candidate_root).summary()[
                        "audit_incomplete"
                    ]
                )

    def test_reopen_flags_incomplete_duplicate_or_out_of_order_revision_chain(self):
        _, events = self.prepare_history(2)
        variants = (
            [events[1]],
            [events[0], events[0], events[1]],
            [events[1], events[0]],
        )

        for variant in variants:
            with self.subTest(revisions=[event["new_revision"] for event in variant]):
                self.write_history(variant)
                self.assertTrue(
                    ReviewStore(self.work_root, self.candidate_root).summary()[
                        "audit_incomplete"
                    ]
                )

    def test_reopen_flags_identity_or_revision_inconsistency(self):
        _, events = self.prepare_history()
        mutations = (
            ("stem", "different_stem"),
            ("prior_revision", 4),
            ("new_revision", 4),
            ("before.stem", "different_stem"),
            ("after.stem", "different_stem"),
            ("before.image_sha256", "a" * 64),
            ("after.image_sha256", "a" * 64),
            ("before.revision", 9),
            ("after.revision", 9),
        )

        for path, value in mutations:
            with self.subTest(path=path):
                event = copy.deepcopy(events[0])
                parts = path.split(".")
                if len(parts) == 1:
                    event[parts[0]] = value
                else:
                    event[parts[0]][parts[1]] = value
                self.write_history([event])
                self.assertTrue(
                    ReviewStore(self.work_root, self.candidate_root).summary()[
                        "audit_incomplete"
                    ]
                )

    def test_reopen_flags_event_before_that_does_not_match_previous_after(self):
        _, events = self.prepare_history(2)
        events[1]["before"]["notes"] = "different prior content"
        self.write_history(events)

        self.assertTrue(
            ReviewStore(self.work_root, self.candidate_root).summary()[
                "audit_incomplete"
            ]
        )

    def test_reopen_flags_nonhuman_event_with_final_or_first_reviewed_after(self):
        store, _, _ = self.prepare(("final", "first_reviewed"))
        store.update("final", self.valid_positive_patch(), 0)
        store.update(
            "first_reviewed",
            {
                "status": "needs_second_review",
                "first_reviewed_at": "2026-07-17T00:00:00+00:00",
                "second_review_required": True,
            },
            0,
        )
        original = self.read_history()
        self.assertFalse(store.summary()["audit_incomplete"])

        for stem in ("final", "first_reviewed"):
            with self.subTest(stem=stem):
                events = copy.deepcopy(original)
                next(event for event in events if event["stem"] == stem)[
                    "origin"
                ] = "model"
                self.write_history(events)
                self.assertTrue(
                    ReviewStore(self.work_root, self.candidate_root).summary()[
                        "audit_incomplete"
                    ]
                )

    def test_reopen_flags_nonhuman_patch_to_previously_final_record(self):
        store, _, _ = self.prepare(("sample_a",))
        store.update("sample_a", self.valid_positive_patch(), 0)
        store.update("sample_a", {"notes": "human post-approval edit"}, 1)
        events = self.read_history()
        self.assertFalse(store.summary()["audit_incomplete"])
        events[1]["origin"] = "system"
        self.write_history(events)

        self.assertTrue(
            ReviewStore(self.work_root, self.candidate_root).summary()[
                "audit_incomplete"
            ]
        )


if __name__ == "__main__":
    unittest.main()
