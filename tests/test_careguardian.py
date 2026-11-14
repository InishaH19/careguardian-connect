import unittest
from pathlib import Path
from src.careguardian import economic_order_quantity, net_present_value, train_and_evaluate
from src.reminder_qa import check_response

ROOT = Path(__file__).resolve().parents[1]

class DemoTests(unittest.TestCase):
    def test_eoq(self):
        self.assertAlmostEqual(economic_order_quantity(100, 8, 4), 20)
    def test_npv(self):
        self.assertAlmostEqual(net_present_value(0, 100, [50, 60]), 10)
    def test_grouped_holdout(self):
        r = train_and_evaluate(ROOT / "data/synthetic_dose_events.csv")
        self.assertGreater(r["test_subjects"], 0)
        self.assertGreaterEqual(r["accuracy"], 0)
        self.assertLessEqual(r["accuracy"], 1)
    def test_unsafe_mock_marker(self):
        self.assertTrue(check_response("double the dose")["manual_review_required"])

if __name__ == "__main__":
    unittest.main()
