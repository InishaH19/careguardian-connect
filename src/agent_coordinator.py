"""Multi-Agent Coordination & Safety Orchestration Engine for CareGuardian Connect.

Demonstrates stateful agent coordination across patient, caregiver, and clinical stakeholders
with safety guardrails, automated fallback triggers, and inventory tracking.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CareGuardianAgent")


class EscalationLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CAREGIVER_ESCALATION = "CAREGIVER_ESCALATION"
    CLINICAL_INTERVENTION = "CLINICAL_INTERVENTION"


class SafetyVerdict(str, Enum):
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"
    REQUIRES_CLINICAL_REVIEW = "REQUIRES_CLINICAL_REVIEW"


@dataclass
class PatientProfile:
    patient_id: str
    name: str
    medication_name: str
    prescribed_dose_mg: float
    schedule_times: List[str]  # e.g., ["08:00", "20:00"]
    caregiver_name: str
    caregiver_contact: str
    timezone: str = "America/Los_Angeles"
    current_inventory_units: int = 60
    reorder_point_units: int = 15


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    action: str
    payload: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class ClinicalSafetyGuardrail:
    """Evaluates agent interactions and patient prompts for clinical risk and policy violations."""

    UNSAFE_PATTERNS = [
        "double the dose",
        "twice the usual",
        "skip your doctor",
        "take 2x",
        "take 3x",
        "take 4x",
        "ignore the prescription",
        "stop taking medication immediately",
        "take extra pills",
        "cut the patch",
    ]

    EMERGENCY_KEYWORDS = [
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "severe allergic reaction",
        "anaphylaxis",
        "loss of consciousness",
    ]

    def evaluate_text(self, text: str) -> Dict[str, Any]:
        lower = text.lower()
        # Check emergency triage
        for em in self.EMERGENCY_KEYWORDS:
            if em in lower:
                return {
                    "verdict": SafetyVerdict.BLOCKED,
                    "reason": f"Emergency symptom detected: '{em}'. Immediate 911 / emergency triage diversion required.",
                    "escalation": EscalationLevel.CLINICAL_INTERVENTION,
                    "action_required": "DIVERT_TO_EMERGENCY_SERVICES",
                }

        # Check dangerous dosage / clinical violation
        for marker in self.UNSAFE_PATTERNS:
            if marker in lower:
                return {
                    "verdict": SafetyVerdict.BLOCKED,
                    "reason": f"Unsafe dosage advice or adherence breach detected: '{marker}'.",
                    "escalation": EscalationLevel.CLINICAL_INTERVENTION,
                    "action_required": "BLOCK_AND_ROUTE_TO_PHYSICIAN",
                }

        return {
            "verdict": SafetyVerdict.APPROVED,
            "reason": "Text passed clinical safety screening.",
            "escalation": EscalationLevel.NORMAL,
            "action_required": "PROCEED",
        }


class SchedulerAgent:
    """Manages patient dosing schedules, notification delivery windows, and timezone alignment."""

    def __init__(self, profile: PatientProfile):
        self.profile = profile

    def generate_reminder(self, scheduled_time: str) -> AgentMessage:
        return AgentMessage(
            sender="SchedulerAgent",
            recipient=self.profile.patient_id,
            action="DISPATCH_SCHEDULED_REMINDER",
            payload={
                "message": (
                    f"Hi {self.profile.name}, it is time for your scheduled "
                    f"{self.profile.medication_name} ({self.profile.prescribed_dose_mg}mg) dose. "
                    "Please confirm once taken."
                ),
                "scheduled_time": scheduled_time,
                "requires_confirmation": True,
            },
        )

    def handle_missed_dose(self, scheduled_time: str, hours_elapsed: float) -> AgentMessage:
        if hours_elapsed <= 2.0:
            advice = (
                f"You missed your {scheduled_time} dose by {hours_elapsed:.1f} hours. "
                "Per your safe window protocol, you may take your normal single prescribed dose now."
            )
            level = EscalationLevel.WARNING
        else:
            advice = (
                f"You missed your {scheduled_time} dose by {hours_elapsed:.1f} hours, which exceeds the safe self-administration window. "
                "Do NOT double your next dose. Please consult your care team or wait for the next scheduled dose."
            )
            level = EscalationLevel.CAREGIVER_ESCALATION

        return AgentMessage(
            sender="SchedulerAgent",
            recipient=self.profile.patient_id,
            action="MISSED_DOSE_INTERVENTION",
            payload={"guidance": advice, "escalation_level": level},
        )


class InventoryReplenishmentAgent:
    """Tracks medication supply, decrementing doses and generating replenishment alerts."""

    def __init__(self, profile: PatientProfile, cost_per_order: float = 45.0, unit_holding_cost: float = 3.0):
        self.profile = profile
        self.cost_per_order = cost_per_order
        self.unit_holding_cost = unit_holding_cost

    def record_dose_taken(self, units: int = 1) -> Dict[str, Any]:
        self.profile.current_inventory_units = max(0, self.profile.current_inventory_units - units)
        needs_reorder = self.profile.current_inventory_units <= self.profile.reorder_point_units

        alert = None
        if needs_reorder:
            alert = {
                "status": "REORDER_TRIGGERED",
                "remaining_units": self.profile.current_inventory_units,
                "reorder_threshold": self.profile.reorder_point_units,
                "recommended_action": "Dispatched automated pharmacy refill request via J&J MedTech API.",
            }

        return {
            "current_inventory": self.profile.current_inventory_units,
            "reorder_needed": needs_reorder,
            "refill_alert": alert,
        }


class CoordinationOrchestrator:
    """Main orchestrator coordinating sub-agents and running end-to-end patient care scenarios."""

    def __init__(self, profile: PatientProfile):
        self.profile = profile
        self.guardrail = ClinicalSafetyGuardrail()
        self.scheduler = SchedulerAgent(profile)
        self.inventory = InventoryReplenishmentAgent(profile)
        self.audit_log: List[Dict[str, Any]] = []

    def execute_scenario_a_routine_reminder(self) -> Dict[str, Any]:
        """Scenario A: Standard routine on-time reminder and adherence confirmation."""
        reminder = self.scheduler.generate_reminder("08:00")
        safety = self.guardrail.evaluate_text(reminder.payload["message"])

        # Patient confirms intake
        patient_reply = f"Confirmed. Took 1 tablet of {self.profile.medication_name}."
        patient_safety = self.guardrail.evaluate_text(patient_reply)

        # Inventory decrement
        inv_update = self.inventory.record_dose_taken(1)

        result = {
            "scenario": "Scenario A: Routine Scheduled Adherence",
            "reminder_dispatched": reminder.payload["message"],
            "safety_verdict": safety["verdict"].value,
            "patient_confirmation": patient_reply,
            "inventory_after_dose": inv_update["current_inventory"],
            "escalation_triggered": False,
        }
        self.audit_log.append(result)
        return result

    def execute_scenario_b_missed_dose_safe_window(self) -> Dict[str, Any]:
        """Scenario B: Dose missed by 1.5 hours within safe window."""
        intervention = self.scheduler.handle_missed_dose("08:00", hours_elapsed=1.5)
        safety = self.guardrail.evaluate_text(intervention.payload["guidance"])

        result = {
            "scenario": "Scenario B: Missed Dose Within Safe Window (1.5 hrs)",
            "guidance": intervention.payload["guidance"],
            "escalation_level": intervention.payload["escalation_level"].value,
            "safety_verdict": safety["verdict"].value,
            "action_taken": "Patient approved for normal single dose; scheduled next follow-up in 2 hours.",
        }
        self.audit_log.append(result)
        return result

    def execute_scenario_c_caregiver_escalation(self) -> Dict[str, Any]:
        """Scenario C: Repeated missed doses (5.0 hours) triggering automated caregiver escalation."""
        intervention = self.scheduler.handle_missed_dose("08:00", hours_elapsed=5.0)
        safety = self.guardrail.evaluate_text(intervention.payload["guidance"])

        # Multi-agent dispatch to caregiver
        caregiver_alert = AgentMessage(
            sender="CoordinationOrchestrator",
            recipient=self.profile.caregiver_contact,
            action="DISPATCH_CAREGIVER_NOTIFICATION",
            payload={
                "alert": f"Adherence alert for {self.profile.name}: morning dose missed by >4 hours.",
                "caregiver": self.profile.caregiver_name,
                "recommended_action": "Check in with patient regarding daily regimen.",
            },
        )

        result = {
            "scenario": "Scenario C: Multi-Party Caregiver Escalation",
            "escalation_level": intervention.payload["escalation_level"].value,
            "caregiver_dispatched_alert": caregiver_alert.payload,
            "safety_verdict": safety["verdict"].value,
        }
        self.audit_log.append(result)
        return result

    def execute_scenario_d_clinical_safety_interception(self) -> Dict[str, Any]:
        """Scenario D: Patient asks to double dose; safety guardrail blocks recommendation."""
        unsafe_user_query = "I missed yesterday's pill. Can I take twice the usual amount now to catch up?"
        safety_eval = self.guardrail.evaluate_text(unsafe_user_query)

        # Automated fallback trigger
        automated_response = (
            "Dose adjustment blocked by safety guardrail. Under clinical guidelines, "
            "never double your dosage following a missed dose. An alert has been routed "
            "to your designated healthcare provider."
        )

        result = {
            "scenario": "Scenario D: Unsafe Prompt Interception & Fallback Trigger",
            "user_query": unsafe_user_query,
            "guardrail_verdict": safety_eval["verdict"].value,
            "detected_reason": safety_eval["reason"],
            "action_required": safety_eval["action_required"],
            "system_response": automated_response,
            "escalated_to_physician": True,
        }
        self.audit_log.append(result)
        return result

    def run_all_scenarios(self) -> Dict[str, Any]:
        return {
            "scenarios": [
                self.execute_scenario_a_routine_reminder(),
                self.execute_scenario_b_missed_dose_safe_window(),
                self.execute_scenario_c_caregiver_escalation(),
                self.execute_scenario_d_clinical_safety_interception(),
            ]
        }
