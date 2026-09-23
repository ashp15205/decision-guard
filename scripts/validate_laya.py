"""
Validation script for the LayaAdapter prompt format.

Run this BEFORE deploying LayaAdapter in production to verify that
the adapter's best-effort prompt format produces meaningful outputs
from the actual convaiinnovations/laya model.

Usage:
    pip install decision-guard[laya]
    python scripts/validate_laya.py

What it does:
    1. Loads the real Laya model from Hugging Face.
    2. Runs a small known dataset of (state, question, expected_answer) triples.
    3. Prints the raw probabilities and whether the top prediction matches.
    4. If accuracy is at random chance (e.g. ~1/N for N options), the prompt
       format is likely wrong and needs to be updated in LayaAdapter._build_prompt.
"""

import sys

try:
    from decision_guard.adapters.laya import LayaAdapter
    from decision_guard.schema import ChoiceQuestion
except ImportError:
    print("ERROR: decision-guard is not installed. Run: pip install -e .[laya]")
    sys.exit(1)

# --- Your test cases here ---
# Format: (state_str, ChoiceQuestion, expected_choice)
TEST_CASES = [
    (
        "The customer's payment failed and they are asking for a refund.",
        ChoiceQuestion(
            id="q_dept",
            description="Route this support ticket to the correct department.",
            options=["billing", "tech_support", "sales", "general"]
        ),
        "billing",
    ),
    (
        "The user cannot log in after resetting their password.",
        ChoiceQuestion(
            id="q_dept2",
            description="Route this support ticket to the correct department.",
            options=["billing", "tech_support", "sales", "general"]
        ),
        "tech_support",
    ),
]


def main():
    print("Loading LayaAdapter (this will download the model if not cached)...")
    try:
        adapter = LayaAdapter()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    print(f"\nRunning {len(TEST_CASES)} validation cases:\n")

    correct = 0
    for i, (state, question, expected) in enumerate(TEST_CASES):
        response = adapter.predict(state, [question])
        answer = response.answers[question.id]
        match = answer.choice == expected
        correct += int(match)

        status = "✅ PASS" if match else "❌ FAIL"
        print(f"[{i+1}] {status}")
        print(f"    State   : {state[:60]}...")
        print(f"    Expected: {expected}")
        print(f"    Got     : {answer.choice} (confidence={answer.confidence:.3f})")
        print()

    accuracy = correct / len(TEST_CASES)
    print(f"Accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.1%}")
    random_baseline = 1 / len(TEST_CASES[0][1].options)
    print(f"Random baseline: {random_baseline:.1%}")

    if accuracy <= random_baseline:
        print("\n⚠️  WARNING: Accuracy is at or below random chance.")
        print("   The prompt format in LayaAdapter._build_prompt is likely incorrect.")
        print("   Check the model card at https://huggingface.co/convaiinnovations/laya")
        print("   and update _build_prompt accordingly.")
    else:
        print("\n✅ Adapter appears functional. Consider applying CalibrationTracker")
        print("   to correct for any remaining over-confidence before production use.")


if __name__ == "__main__":
    main()
