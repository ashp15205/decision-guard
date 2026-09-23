import math
from typing import List, Dict, Tuple, Any
from decision_guard.store import LocalStore
from decision_guard.schema import PredictionResponse

class CalibrationTracker:
    def __init__(self, store: LocalStore):
        self.store = store
        self.temperature_params: Dict[str, Tuple[float, float]] = {}  # question_id -> (T, B)

    def compute_ece(self, question_id: str, n_bins: int = 10) -> float:
        """
        Computes the Expected Calibration Error (ECE) for a given question.
        Returns 0.0 if there is no data.
        """
        records = self.store.load_all_records()
        
        # Filter records for this question that have both prediction and outcome
        relevant = [
            r for r in records.values()
            if r.get("question_id") == question_id 
            and "prediction" in r 
            and "outcome" in r
            and "confidence" in r
        ]
        
        if not relevant:
            return 0.0
            
        bins: Dict[int, List[Dict]] = {i: [] for i in range(n_bins)}
        
        for r in relevant:
            conf = r["confidence"]
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append(r)
            
        ece = 0.0
        total_samples = len(relevant)
        
        for bin_idx, items in bins.items():
            if not items:
                continue
                
            avg_conf = sum(i["confidence"] for i in items) / len(items)
            # Accuracy: 1 if prediction matches outcome, else 0
            acc = sum(1 for i in items if i["prediction"] == i["outcome"]) / len(items)
            
            weight = len(items) / total_samples
            ece += weight * abs(avg_conf - acc)
            
        return ece

    def _apply_temperature(self, conf: float, t: float, b: float) -> float:
        """Applies Platt scaling (temperature + bias) to a single confidence value."""
        # Clamp conf to prevent math domain errors
        conf = max(min(conf, 0.999999), 0.000001)
        
        # Derived logit
        logit = math.log(conf / (1 - conf))
        scaled_logit = (logit / t) + b
        
        # Sigmoid to get back to probability
        return 1 / (1 + math.exp(-scaled_logit))

    def fit_temperature(self, question_id: str):
        """
        Fits a temperature parameter T and bias B to minimize Negative Log Likelihood (NLL)
        for a given question, using a simple grid search.
        """
        records = self.store.load_all_records()
        relevant = [
            r for r in records.values()
            if r.get("question_id") == question_id 
            and "prediction" in r 
            and "outcome" in r
            and "confidence" in r
        ]
        
        if not relevant:
            return

        best_t = 1.0
        best_b = 0.0
        min_nll = float('inf')
        
        # Grid search T from 0.1 to 10.0, and B from -10.0 to 5.0
        for t_int in range(1, 101, 5):
            t = t_int / 10.0
            for b_int in range(-100, 51, 5):
                b = b_int / 10.0
                
                nll = 0.0
                for r in relevant:
                    conf = r["confidence"]
                    scaled_conf = self._apply_temperature(conf, t, b)
                    
                    # NLL calculation
                    is_correct = (r["prediction"] == r["outcome"])
                    # Clamp to avoid log(0)
                    scaled_conf = max(min(scaled_conf, 0.9999), 0.0001)
                    
                    if is_correct:
                        nll -= math.log(scaled_conf)
                    else:
                        nll -= math.log(1 - scaled_conf)
                
                if nll < min_nll:
                    min_nll = nll
                    best_t = t
                    best_b = b
                    
        self.temperature_params[question_id] = (best_t, best_b)

    def calibrated_predict(self, response: PredictionResponse) -> PredictionResponse:
        """
        Takes a raw PredictionResponse and applies the learned temperature/bias scaling
        to the confidences.
        """
        for q_id, answer in response.answers.items():
            t, b = self.temperature_params.get(q_id, (1.0, 0.0))
            if t != 1.0 or b != 0.0:
                answer.confidence = self._apply_temperature(answer.confidence, t, b)
                
        return response
