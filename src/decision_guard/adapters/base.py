import abc
from typing import List, Any
from decision_guard.schema import Question, PredictionResponse

class BaseAdapter(abc.ABC):
    """Abstract base class for all decision model adapters."""
    
    @abc.abstractmethod
    def predict(self, state: Any, questions: List[Question]) -> PredictionResponse:
        """
        Execute a prediction against the underlying model.
        
        Args:
            state: The unstructured context or state to evaluate.
            questions: A list of Question definitions (Choice, Score, Noul).
            
        Returns:
            A PredictionResponse containing answers and confidence scores.
        """
        pass
