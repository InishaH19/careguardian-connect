"""Tests for Multi-Agent Coordinator and Clinical Safety Guardrails."""
import unittest
from src.agent_coordinator import (
    CoordinationOrchestrator,
    PatientProfile,
    ClinicalSafetyGuardrail,
    SafetyVerdict,
    EscalationLevel,
)


class TestAgentCoordinator(unittest.TestCase):
    def setUp(self):
        self.profile = PatientProfile(
            patient_id="PT-TEST-01",
            name="Test User",
            medication_name="TestMed",
            prescribed_dose_mg=50.0,
            schedule_times=["09:00", "21:00"],
            caregiver_name="Test Caregiver",
            caregiver_contact="+1-555-0100",
            current_inventory_units=30,
            reorder_point_units=10,
        )
        self.orchestrator = CoordinationOrchestrator(self.profile)
        self.guardrail = ClinicalSafetyGuardrail()

    def test_scenario_a_routine_reminder(self):
        result = self.orchestrator.execute_scenario_a_routine_reminder()
        self.assertEqual(result["safety_verdict"], SafetyVerdict.APPROVED.value)
        self.assertEqual(result["inventory_after_dose"], 29)
        self.assertFalse(result["escalation_triggered"])

    def test_scenario_b_missed_dose_safe_window(self):
        result = self.orchestrator.execute_scenario_b_missed_dose_safe_window()
        self.assertEqual(result["escalation_level"], EscalationLevel.WARNING.value)
        self.assertEqual(result["safety_verdict"], SafetyVerdict.APPROVED.value)

    def test_scenario_c_caregiver_escalation(self):
        result = self.orchestrator.execute_scenario_c_caregiver_escalation()
        self.assertEqual(result["escalation_level"], EscalationLevel.CAREGIVER_ESCALATION.value)
        self.assertIn("Adherence alert", result["caregiver_dispatched_alert"]["alert"])

    def test_scenario_d_clinical_safety_interception(self):
        result = self.orchestrator.execute_scenario_d_clinical_safety_interception()
        self.assertEqual(result["guardrail_verdict"], SafetyVerdict.BLOCKED.value)
        self.assertTrue(result["escalated_to_physician"])
        self.assertEqual(result["action_required"], "BLOCK_AND_ROUTE_TO_PHYSICIAN")

    def test_emergency_triage_detection(self):
        eval_res = self.guardrail.evaluate_text("I have acute chest pain and dizziness")
        self.assertEqual(eval_res["verdict"], SafetyVerdict.BLOCKED)
        self.assertEqual(eval_res["action_required"], "DIVERT_TO_EMERGENCY_SERVICES")

    def test_safe_query_passes(self):
        eval_res = self.guardrail.evaluate_text("Can I take my pill with orange juice?")
        self.assertEqual(eval_res["verdict"], SafetyVerdict.APPROVED)
        self.assertEqual(eval_res["action_required"], "PROCEED")


if __name__ == "__main__":
    unittest.main()
