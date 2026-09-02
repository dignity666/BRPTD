"""原子制品与 JSON 禁止 NaN 的回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brptd.artifacts import ArtifactError, write_csv_records, write_manifest


class ArtifactTests(unittest.TestCase):
    def test_manifest_is_stable_and_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.json"
            write_manifest(target, {"b": [1, 2], "a": 3})
            first = target.read_bytes()
            write_manifest(target, {"b": [1, 2], "a": 3})
            self.assertEqual(first, target.read_bytes())
            with self.assertRaises(ArtifactError):
                write_manifest(target, {"bad": float("nan")})
            self.assertEqual(first, target.read_bytes())

    def test_csv_column_order_null_and_failed_write_preserve_old_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trial.csv"
            write_csv_records(target, ("a", "b"), ({"b": None, "a": 1.25},))
            before = target.read_text(encoding="utf-8")
            self.assertEqual("a,b\n1.25,\n", before)
            with self.assertRaises(ArtifactError):
                write_csv_records(target, ("a", "b"), ({"a": 1},))
            self.assertEqual(before, target.read_text(encoding="utf-8"))

    def test_artifact_rejects_unsupported_nested_values_and_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact.json"
            with self.assertRaises(ArtifactError):
                write_manifest(target, {"unsupported": {1, 2}})
            with self.assertRaises(ArtifactError):
                write_manifest(target, {"nested": [float("inf")]})
            target.write_text("old", encoding="utf-8")
            with self.assertRaises(ArtifactError):
                write_csv_records(target, ("a",), ({"a": 1, "b": 2},))
            self.assertEqual("old", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
