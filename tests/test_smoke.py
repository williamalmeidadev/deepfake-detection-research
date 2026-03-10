import csv
from pathlib import Path
import py_compile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestSmoke(unittest.TestCase):
    def test_python_files_compile(self) -> None:
        files = [
            ROOT / "notebook" / "app.py",
            ROOT / "scripts" / "generate_timeseries.py",
            ROOT / "scripts" / "run_pca.py",
            ROOT / "scripts" / "train_classifier.py",
            ROOT / "scripts" / "train_prophet.py",
        ]
        for file_path in files:
            self.assertTrue(file_path.exists(), f"Missing file: {file_path}")
            py_compile.compile(str(file_path), doraise=True)

    def test_processed_csvs_have_expected_columns(self) -> None:
        expected = {
            ROOT / "data" / "processed" / "deepfake_dataset_cleaned.csv": {
                "media_type",
                "content_category",
                "face_count",
                "audio_present",
                "lip_sync_score",
                "visual_artifacts_score",
                "compression_level",
                "lighting_inconsistency_score",
                "source_platform",
                "label",
            },
            ROOT / "data" / "processed" / "df_timeseries.csv": {"Data", "Volume_Deepfakes"},
            ROOT / "data" / "processed" / "prophet_forecast.csv": {"ds", "yhat"},
        }

        for csv_path, required_cols in expected.items():
            self.assertTrue(csv_path.exists(), f"Missing dataset: {csv_path}")
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertIsNotNone(reader.fieldnames, f"No header in: {csv_path}")
                header = set(reader.fieldnames or [])
                missing = required_cols - header
                self.assertFalse(missing, f"Missing columns in {csv_path}: {sorted(missing)}")


if __name__ == "__main__":
    unittest.main()
