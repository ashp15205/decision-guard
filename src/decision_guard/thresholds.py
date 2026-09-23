from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel
from decision_guard.store import LocalStore
from decision_guard.schema import Answer

class GateDecision(str, Enum):
    ACT = "act"
    ESCALATE_HUMAN = "escalate_human"
    ESCALATE_LLM = "escalate_llm"

class CostProfile(BaseModel):
    fp_cost: float
    fn_cost: float

class ThresholdManager:
    def __init__(self, store: LocalStore):
        self.store = store
        self.cost_profiles: Dict[str, CostProfile] = {}
        self.recommended_thresholds: Dict[str, float] = {}
        # Default global threshold if we don't have enough data or no profile set
        self.default_threshold = 0.8
        
    def set_cost_profile(self, question_id: str, fp_cost: float, fn_cost: float):
        self.cost_profiles[question_id] = CostProfile(fp_cost=fp_cost, fn_cost=fn_cost)
        
    def compute_recommended_threshold(self, question_id: str, target_answer: Any) -> Optional[float]:
        """
        Computes the optimal confidence threshold that minimizes total cost, 
        assuming `target_answer` is the "positive" case (e.g. Action: "refund").
        """
        if question_id not in self.cost_profiles:
            return None
            
        profile = self.cost_profiles[question_id]
        records = self.store.load_all_records()
        
        relevant = [
            r for r in records.values()
            if r.get("question_id") == question_id 
            and "prediction" in r 
            and "outcome" in r
            and "confidence" in r
        ]
        
        if len(relevant) < 10:
            return None  # Insufficient data to make a recommendation
            
        best_threshold = 0.5
        min_cost = float('inf')
        
        # Grid search threshold from 0.0 to 1.0
        for th_int in range(0, 101):
            th = th_int / 100.0
            cost = 0.0
            
            for r in relevant:
                pred = r["prediction"]
                conf = r["confidence"]
                actual = r["outcome"]
                
                # If we act (confidence >= th and prediction == target_answer)
                act = (conf >= th) and (pred == target_answer)
                
                if act and actual != target_answer:
                    # False Positive: We acted, but we shouldn't have
                    cost += profile.fp_cost
                elif not act and actual == target_answer:
                    # False Negative: We escalated, but we should have acted
                    # (Escalation itself is the cost here)
                    cost += profile.fn_cost
                    
            if cost < min_cost:
                min_cost = cost
                best_threshold = th
                
        # We store the recommended threshold for the target_answer
        key = f"{question_id}:{target_answer}"
        self.recommended_thresholds[key] = best_threshold
        return best_threshold

    def gate(self, answer: Answer, target_answer: Any, llm_fallback: bool = True) -> GateDecision:
        """
        Decides whether to act or escalate based on the recommended threshold.
        """
        key = f"{answer.question_id}:{target_answer}"
        th = self.recommended_thresholds.get(key, self.default_threshold)
        
        # We only act if the model actually predicted the target_answer AND confidence is high enough
        # Wait, the spec says "returns act / escalate-to-human / escalate-to-LLM".
        # If the model didn't predict the target answer, we effectively "act" by doing nothing (or the opposite).
        # For simplicity, let's assume `gate()` is called to verify a specific high-stakes prediction.
        
        # NOTE: A more robust implementation would handle multi-class better.
        prediction_val = None
        if hasattr(answer, 'choice'):
            prediction_val = answer.choice
        elif hasattr(answer, 'score'):
            prediction_val = answer.score
        elif hasattr(answer, 'result'):
            prediction_val = answer.result
            
        if prediction_val == target_answer and answer.confidence >= th:
            return GateDecision.ACT
            
        # If it doesn't meet the threshold, escalate
        return GateDecision.ESCALATE_LLM if llm_fallback else GateDecision.ESCALATE_HUMAN
