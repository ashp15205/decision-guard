import os
import pytest
from decision_guard.store import LocalStore
from decision_guard.calibration import CalibrationTracker
from decision_guard.schema import PredictionResponse, ChoiceAnswer

def test_laya_11_plus_option_bug():
    """
    Validates that the calibration module fixes the real Laya 11+ option bug:
    Where choice questions with >= 11 options saturate at 1.0 confidence
    but are frequently wrong.
    """
    store_path = "test_laya_bug.jsonl"
    if os.path.exists(store_path):
        os.remove(store_path)
        
    store = LocalStore(store_path)
    tracker = CalibrationTracker(store)
    question_id = "q_12_options"
    
    # 1. Generate Buggy Data (High confidence, low accuracy)
    # Simulate 100 predictions. All have 1.0 confidence, but only 10% are correct.
    for i in range(100):
        pred_id = store.log_prediction(
            question_id=question_id,
            state={"context": "dummy"},
            prediction="option_A",
            confidence=1.0  # The Laya bug: silently saturates to 1.0
        )
        
        # Real outcome: 90% of the time it's wrong
        real_outcome = "option_A" if i < 10 else "option_B"
        store.log_outcome(pred_id, question_id, real_outcome)
        
    # 2. Check initial ECE
    initial_ece = tracker.compute_ece(question_id)
    # The average confidence is 1.0, the accuracy is 0.1, so ECE should be around 0.9.
    assert initial_ece > 0.8, f"Initial ECE should be high, got {initial_ece}"
    
    # 3. Fit Temperature
    tracker.fit_temperature(question_id)
    assert tracker.temperature_params[question_id] != 1.0, "Temperature scaling should adjust the parameter"
    
    # 4. Apply Calibration and re-evaluate
    # We create a dummy response that exhibits the bug
    response = PredictionResponse(
        answers={
            question_id: ChoiceAnswer(question_id=question_id, confidence=1.0, choice="option_A")
        }
    )
    
    calibrated = tracker.calibrated_predict(response)
    new_confidence = calibrated.answers[question_id].confidence
    
    # The new confidence should be much lower, closer to the real accuracy of 0.1
    assert new_confidence < 0.5, f"Calibrated confidence {new_confidence} is still too high."
    
    # Cleanup
    if os.path.exists(store_path):
        os.remove(store_path)
