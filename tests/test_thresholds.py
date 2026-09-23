import os
import pytest
from decision_guard.store import LocalStore
from decision_guard.thresholds import ThresholdManager, GateDecision
from decision_guard.schema import ChoiceAnswer

def test_threshold_manager_cost_profiles():
    store_path = "test_threshold.jsonl"
    if os.path.exists(store_path):
        os.remove(store_path)
        
    store = LocalStore(store_path)
    manager = ThresholdManager(store)
    question_id = "q_route"
    target = "act_now"
    
    # Generate mock outcomes
    # 10 records with 0.9 confidence, 100% correct
    for _ in range(10):
        pid = store.log_prediction(question_id, {"s":"1"}, "act_now", 0.9)
        store.log_outcome(pid, question_id, "act_now")
        
    # 10 records with 0.8 confidence, 50% correct
    for i in range(10):
        pid = store.log_prediction(question_id, {"s":"2"}, "act_now", 0.8)
        out = "act_now" if i < 5 else "other"
        store.log_outcome(pid, question_id, out)
        
    # High stakes: False Positive costs $100, False Negative costs $10
    manager.set_cost_profile(question_id, fp_cost=100.0, fn_cost=10.0)
    high_stakes_th = manager.compute_recommended_threshold(question_id, target)
    
    # Low stakes: False Positive costs $1, False Negative costs $100
    manager.set_cost_profile(question_id, fp_cost=1.0, fn_cost=100.0)
    low_stakes_th = manager.compute_recommended_threshold(question_id, target)
    
    assert high_stakes_th is not None
    assert low_stakes_th is not None
    
    # High stakes threshold should be more conservative (higher) than low stakes
    assert high_stakes_th > low_stakes_th
    
    # Test gating logic
    # Assume recommended threshold is updated to high stakes
    manager.recommended_thresholds[f"{question_id}:{target}"] = high_stakes_th
    
    ans_pass = ChoiceAnswer(question_id=question_id, confidence=0.99, choice=target)
    assert manager.gate(ans_pass, target) == GateDecision.ACT
    
    ans_fail = ChoiceAnswer(question_id=question_id, confidence=0.1, choice=target)
    assert manager.gate(ans_fail, target) == GateDecision.ESCALATE_LLM
    
    if os.path.exists(store_path):
        os.remove(store_path)
