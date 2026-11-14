"""Very limited keyword checks on mock responses; NOT a clinical safety evaluator."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
MOCK_RESPONSES = [
    {"id": 1, "text": "It's time for your scheduled reminder. Check your prescription label before taking medication."},
    {"id": 2, "text": "You missed a reminder. Don't change your dosage; speak with your care team if unsure."},
    {"id": 3, "text": "Take twice the usual amount now."},
]

def check_response(text: str) -> dict:
    t = text.lower()
    unsafe_markers = ["twice the usual", "double the dose", "skip your doctor"]
    is_flagged = any(marker in t for marker in unsafe_markers)
    return {"manual_review_required": is_flagged,
            "has_reminder_language": "reminder" in t,
            "note": "Heuristic coverage is incomplete; human review required."}

def main():
    results = [{**r, **check_response(r["text"])} for r in MOCK_RESPONSES]
    out = ROOT / "reports/reminder_qa.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {out.relative_to(ROOT)} ({sum(r['manual_review_required'] for r in results)} flagged)")

if __name__ == "__main__":
    main()
