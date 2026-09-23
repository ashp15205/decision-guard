from typing import List, Union, Literal, Any, Dict, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Question Definitions (Input to the model)
# ---------------------------------------------------------------------------

class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    id: str
    description: str
    options: List[str]

class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    id: str
    description: str
    min_score: int
    max_score: int

class NoulQuestion(BaseModel):
    """Noul is a boolean / probability of truth."""
    type: Literal["noul"] = "noul"
    id: str
    description: str

Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]

# ---------------------------------------------------------------------------
# Answer Definitions (Output from the model)
# ---------------------------------------------------------------------------

class Answer(BaseModel):
    question_id: str
    confidence: float

class ChoiceAnswer(Answer):
    type: Literal["choice"] = "choice"
    choice: str

class ScoreAnswer(Answer):
    type: Literal["score"] = "score"
    score: int

class NoulAnswer(Answer):
    type: Literal["noul"] = "noul"
    result: bool

class PredictionResponse(BaseModel):
    """The standard response format returned by all adapters."""
    answers: Dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]]

