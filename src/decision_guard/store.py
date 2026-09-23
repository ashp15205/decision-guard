import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class LogEntry(BaseModel):
    id: str
    timestamp: str
    event_type: str  # "prediction" or "outcome"
    question_id: str
    
    # For prediction events
    state_hash: Optional[str] = None
    prediction: Optional[Any] = None
    confidence: Optional[float] = None
    
    # For outcome events
    real_outcome: Optional[Any] = None

class LocalStore:
    """
    Append-only JSONL local logging store.
    Records predictions and later outcomes without requiring a database.
    """
    def __init__(self, filepath: str = "decision_guard_logs.jsonl"):
        self.filepath = Path(filepath)
        # Ensure file exists
        self.filepath.touch(exist_ok=True)
        
    def _hash_state(self, state: Any) -> str:
        state_str = json.dumps(state, sort_keys=True) if isinstance(state, (dict, list)) else str(state)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def log_prediction(self, question_id: str, state: Any, prediction: Any, confidence: float) -> str:
        """
        Logs a prediction event. Returns a unique ID for this prediction so an outcome can be attached later.
        """
        entry_id = str(uuid.uuid4())
        entry = LogEntry(
            id=entry_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type="prediction",
            question_id=question_id,
            state_hash=self._hash_state(state),
            prediction=prediction,
            confidence=confidence
        )
        
        with open(self.filepath, "a") as f:
            f.write(entry.model_dump_json() + "\n")
            
        return entry_id

    def log_outcome(self, prediction_id: str, question_id: str, real_outcome: Any):
        """
        Logs the real outcome for a previously recorded prediction.
        """
        entry = LogEntry(
            id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type="outcome",
            question_id=question_id,
            real_outcome=real_outcome
        )
        
        with open(self.filepath, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def load_all_records(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads all JSONL records into memory, merging predictions with their outcomes based on ID.
        Returns: {prediction_id: {"prediction": ..., "confidence": ..., "outcome": ..., "question_id": ...}}
        """
        records = {}
        with open(self.filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                entry_id = data["id"]
                
                if entry_id not in records:
                    records[entry_id] = {}
                    
                if data["event_type"] == "prediction":
                    records[entry_id].update({
                        "question_id": data["question_id"],
                        "prediction": data["prediction"],
                        "confidence": data["confidence"],
                        "state_hash": data["state_hash"],
                        "timestamp": data["timestamp"]
                    })
                elif data["event_type"] == "outcome":
                    records[entry_id]["outcome"] = data["real_outcome"]
                    
        return records
