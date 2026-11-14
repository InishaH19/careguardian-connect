"""Tests for CSV validation and the end-to-end reproducible demo."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from src.adherence_analysis import load_events, descriptive_summary
from src.run_pipeline import run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic_dose_events.csv"

class DataTests(unittest.TestCase):
    def test_valid_dataset(self):
        df = load_events(DATA)
        summary = descriptive_summary(df)
        self.assertEqual(summary["total_events"], len(df))
        self.assertGreater(summary["simulated_participants"], 5)
        self.assertTrue(0 <= summary["observed_completion_pct"] <= 100)

    def test_duplicate_participant_day_rejected(self):
        with DATA.open(newline="") as source:
            rows = list(csv.DictReader(source))
        rows.append(rows[0].copy())
        with TemporaryDirectory() as td:
            path = Path(td) / "broken.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_events(path)

    def test_all_report_files_generated(self):
        with TemporaryDirectory() as td:
            result = run(DATA, Path(td))
            self.assertGreaterEqual(result["rows"], 50)
            for name in ("adherence_summary.json", "model_metrics.json", "economics.json",
                         "agent_coordination_scenarios.json", "safety_eval_matrix.json",
                         "supply_continuity_report.json", "reminder_qa.json", "PROJECT_REPORT.md"):
                self.assertTrue((Path(td) / name).exists(), name)
            report = json.loads((Path(td) / "model_metrics.json").read_text())
            self.assertGreater(report["test_subjects"], 0)

if __name__ == "__main__":
    unittest.main()
