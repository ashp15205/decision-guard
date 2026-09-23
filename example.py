import json
from decision_guard.schema import ChoiceQuestion
from decision_guard.adapters.laya import LayaAdapter
from decision_guard.adapters.jev import JevAdapter
from decision_guard.store import LocalStore
from decision_guard.calibration import CalibrationTracker
from decision_guard.thresholds import ThresholdManager, GateDecision
from decision_guard.safety import StateSafetyGuard

def run_example():
    print("--- Running decision-guard end-to-end example ---")
    
    # 1. Initialize Adapters
    # Laya requires torch/transformers. Jev is a pure HTTP wrapper.
    # We will use JevAdapter here to demonstrate as it requires zero setup to mock.
    print("\n1. Initializing Adapter...")
    adapter = JevAdapter(api_key="mock_key")
    # adapter = LayaAdapter() # If you have the optional dependencies installed
    
    # 2. Define State and Question
    state = "The customer wants to reset their password but the email link is expired."
    questions = [
        ChoiceQuestion(
            id="q_routing",
            description="Route this support ticket to the correct department.",
            options=["billing", "tech_support", "sales", "general"]
        )
    ]
    
    # 3. Safety Guard
    print("\n2. Running Safety Guard...")
    safety = StateSafetyGuard()
    scan_result = safety.scan_state(state)
    if not scan_result.is_safe:
        print(f"  [ERROR] Injection detected: {scan_result.flagged_patterns}")
        return
    print("  [OK] State is safe.")
    
    # Optional: Run Adversarial Test Harness
    print("  [INFO] Running Adversarial Test Harness...")
    adv_results = safety.test_adversarial(
        adapter, 
        questions, 
        base_state=state, 
        expected_answers={"q_routing": "tech_support"} # assuming mock returns 'billing' (option 0)
    )
    print(f"  [INFO] Adversarial tests flipped answers: {any(r['flipped'] for r in adv_results.values())}")

    # 4. Prediction
    print("\n3. Executing Prediction...")
    response = adapter.predict(state, questions)
    raw_answer = response.answers["q_routing"]
    print(f"  Raw Answer: {raw_answer.choice}, Confidence: {raw_answer.confidence:.2f}")
    
    # 5. Calibration
    print("\n4. Applying Calibration...")
    store = LocalStore("example_logs.jsonl")
    tracker = CalibrationTracker(store)
    
    # Log the prediction
    pred_id = store.log_prediction(
        question_id="q_routing",
        state=state,
        prediction=raw_answer.choice,
        confidence=raw_answer.confidence
    )
    
    # Mocking that we previously fitted a temperature T = 1.5
    tracker.temperature_params["q_routing"] = 1.5
    calibrated_response = tracker.calibrated_predict(response)
    cal_answer = calibrated_response.answers["q_routing"]
    print(f"  Calibrated Confidence: {cal_answer.confidence:.2f}")
    
    # Log the outcome later (simulating human feedback)
    store.log_outcome(pred_id, "q_routing", "tech_support")
    
    # 6. Threshold Management
    print("\n5. Threshold Gating...")
    manager = ThresholdManager(store)
    # Set high cost for false positives, lower for false negatives
    manager.set_cost_profile("q_routing", fp_cost=100.0, fn_cost=10.0)
    
    # In a real scenario, this computes based on historical logs
    # For the example, we'll force the recommended threshold
    manager.recommended_thresholds["q_routing:tech_support"] = 0.90
    
    decision = manager.gate(cal_answer, target_answer="tech_support")
    print(f"  Gate Decision: {decision.value}")
    if decision == GateDecision.ACT:
        print("  -> Executing action automatically.")
    else:
        print("  -> Escalating to LLM/Human for review.")

if __name__ == "__main__":
    run_example()
