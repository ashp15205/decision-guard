from decision_guard.schema import ChoiceQuestion
from decision_guard.adapters.jev import JevAdapter
from decision_guard.safety import StateSafetyGuard
from decision_guard.calibration import CalibrationTracker
from decision_guard.thresholds import ThresholdManager, GateDecision
from decision_guard.store import LocalStore

def run_example():
    print("--- Running decision-guard end-to-end example ---\n")
    
    # 1. Setup your tools
    store = LocalStore("logs.jsonl")
    safety_guard = StateSafetyGuard()
    tracker = CalibrationTracker(store)
    thresholds = ThresholdManager(store)
    
    # We will try to load the LayaAdapter. If torch/transformers aren't installed,
    # we fall back to a mocked JevAdapter so the script always runs out of the box.
    try:
        from decision_guard.adapters.laya import LayaAdapter
        print("[INFO] Loading LayaAdapter (this may take a moment to download weights)...")
        adapter = LayaAdapter()
    except ImportError:
        print("[INFO] 'torch' and 'transformers' not installed. Falling back to mocked JevAdapter.")
        adapter = JevAdapter(api_key="mock_key")
    
    # 2. Define the decision you need the model to make
    question = ChoiceQuestion(
        id="q_routing",
        description="Route this support ticket to the correct department.",
        options=["billing", "tech_support", "sales", "general"]
    )
    user_input = "I need a refund for my last purchase."
    print(f"\nUser Input: '{user_input}'")
    
    # 3. Scan for adversarial injections BEFORE calling the model
    scan = safety_guard.scan_state(user_input)
    if not scan.is_safe:
        raise ValueError(f"Injection detected: {scan.flagged_patterns}")
    print("[OK] Input is safe. No adversarial patterns detected.")
    
    # 4. Execute the prediction
    print("[INFO] Running prediction...")
    raw_response = adapter.predict(user_input, [question])
    raw_answer = raw_response.answers["q_routing"]
    print(f"  -> Raw Prediction: {raw_answer.choice} (Confidence: {raw_answer.confidence:.2f})")
    
    # 5. Calibrate the over-confident raw scores based on historical accuracy
    # (Mocking that we previously fitted a temperature T=1.5 to correct overconfidence)
    tracker.temperature_params["q_routing"] = (1.5, 0.0) 
    calibrated_response = tracker.calibrated_predict(raw_response)
    answer = calibrated_response.answers["q_routing"]
    print(f"  -> Calibrated Confidence: {answer.confidence:.2f}")
    
    # 6. Make a safe decision based on the financial cost of a mistake
    thresholds.set_cost_profile(question_id="q_routing", fp_cost=1000.0, fn_cost=10.0)
    
    # Force recommendation threshold for the sake of the example
    thresholds.recommended_thresholds["q_routing:billing"] = 0.85
    
    decision = thresholds.gate(answer, target_answer="billing")
    
    print("\n--- Final Decision ---")
    if decision == GateDecision.ACT:
        print("Confidence is high enough. Routing to billing automatically.")
    else:
        print("Confidence is too low for the cost of a mistake. Escalating to human.")

if __name__ == "__main__":
    run_example()
