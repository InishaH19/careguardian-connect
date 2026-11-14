"""Tests for Adversarial Safety Matrix and Inventory Continuity Modeling."""
import unittest
from src.adversarial_safety_eval import AdversarialSafetyEvaluator, ADVERSARIAL_TEST_MATRIX
from src.inventory_continuity import (
    calculate_eoq,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_npv,
    run_supply_continuity_simulation,
)


class TestSafetyAndSupply(unittest.TestCase):
    def test_adversarial_matrix_pass_rate(self):
        evaluator = AdversarialSafetyEvaluator()
        summary = evaluator.evaluate_all()
        self.assertEqual(summary["total_adversarial_tests"], len(ADVERSARIAL_TEST_MATRIX))
        self.assertEqual(summary["passed_tests"], len(ADVERSARIAL_TEST_MATRIX))
        self.assertEqual(summary["pass_rate_percentage"], 100.0)
        self.assertEqual(summary["status"], "ALL_TESTS_PASSED")

    def test_eoq_calculation(self):
        # D=36000, S=45, H=3 -> sqrt(2 * 36000 * 45 / 3) = sqrt(1080000) = 1039.23
        eoq = calculate_eoq(36000, 45, 3)
        self.assertAlmostEqual(eoq, 1039.23, places=2)

    def test_eoq_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calculate_eoq(-100, 45, 3)

    def test_safety_stock(self):
        # Z=1.645, sigma=10, L=4 -> 1.645 * 10 * 2 = 32.9
        ss = calculate_safety_stock(1.645, 10.0, 4.0)
        self.assertAlmostEqual(ss, 32.9, places=1)

    def test_reorder_point(self):
        # d=100, L=5, SS=30 -> 500 + 30 = 530
        rop = calculate_reorder_point(100.0, 5.0, 30.0)
        self.assertEqual(rop, 530.0)

    def test_npv_calculation(self):
        # rate=0, initial=100, benefits=[50, 60] -> -100 + 110 = 10
        npv = calculate_npv(0.0, 100.0, [50.0, 60.0])
        self.assertEqual(npv, 10.0)

    def test_supply_continuity_simulation_report(self):
        report = run_supply_continuity_simulation()
        self.assertIn("inventory_optimization", report)
        self.assertIn("supply_resilience_and_stockout_prevention", report)
        self.assertIn("financial_feasibility_npv", report)
        self.assertEqual(report["financial_feasibility_npv"]["roi_verdict"], "FINANCIALLY_ACCRETIVE")


if __name__ == "__main__":
    unittest.main()
