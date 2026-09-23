"""
LayaAdapter: Local inference adapter for ConvAI Innovations' Laya model.

Laya is a ModernBERT-based sequence classification model hosted at:
  https://huggingface.co/convaiinnovations/laya

IMPORTANT: The exact prompt format and label mapping for this model has not
been publicly documented in the model card at time of writing. The adapter
below implements a best-effort BERT-style NLI prompt format:

  "[CLS] {state} [SEP] {question_description} [OPTIONS] {opt1} | {opt2} | ... [SEP]"

This format is a reasonable assumption for a multi-class classifier. If you
have access to the exact tokenization format from ConvAI Innovations, update
the `_build_prompt` method accordingly.

Validation: Before deploying this in production, run the provided
`scripts/validate_laya.py` script to compare the adapter's raw outputs
against expected responses on a known dataset. It will help you detect
if the prompt format needs adjustment.
"""
import json
from typing import List, Any

from decision_guard.adapters.base import BaseAdapter
from decision_guard.schema import Question, PredictionResponse, ChoiceAnswer, ScoreAnswer, NoulAnswer


class LayaAdapter(BaseAdapter):
    """
    Adapter for running Laya locally via Hugging Face transformers.

    Requires the `[laya]` optional extra:
      pip install decision-guard[laya]

    Args:
        model_name: Hugging Face model ID. Defaults to 'convaiinnovations/laya'.
        device: 'cpu', 'cuda', or 'mps'. Auto-detected if not specified.
    """

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "LayaAdapter requires torch and transformers. "
                "Install them with: pip install decision-guard[laya]"
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    def _build_prompt(self, state_str: str, question: "Question") -> str:
        """
        Build the input prompt for the Laya classifier.

        NOTE: This format is an informed assumption based on standard BERT NLI
        patterns. Validate against the model card or ConvAI documentation before
        production use. Update this method if a different format is required.
        """
        if question.type == "choice":
            options_str = " | ".join(question.options)
            return f"{state_str} [SEP] {question.description} [OPTIONS] {options_str}"
        else:
            return f"{state_str} [SEP] {question.description}"

    def predict(self, state: Any, questions: List["Question"]) -> PredictionResponse:
        """
        Run local inference using the Laya model.

        The raw logits from the model are converted to probabilities via
        softmax. Because Laya's internal label-to-class mapping is not
        publicly documented, we map the highest-probability class index
        to the corresponding option (for Choice) or a binary result (for
        Noul). Temperature scaling via CalibrationTracker is strongly
        recommended on top of these raw outputs.
        """
        torch = self.torch
        state_str = json.dumps(state) if not isinstance(state, str) else state
        answers = {}

        for q in questions:
            prompt = self._build_prompt(state_str, q)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

            max_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            confidence = float(probs[max_idx])

            if q.type == "choice":
                # Guard against index overflow if num_labels < num_options
                choice_idx = max_idx if max_idx < len(q.options) else 0
                answers[q.id] = ChoiceAnswer(
                    question_id=q.id,
                    confidence=confidence,
                    choice=q.options[choice_idx],
                )
            elif q.type == "score":
                score_range = q.max_score - q.min_score + 1
                score_idx = max_idx if max_idx < score_range else 0
                answers[q.id] = ScoreAnswer(
                    question_id=q.id,
                    confidence=confidence,
                    score=q.min_score + score_idx,
                )
            elif q.type == "noul":
                answers[q.id] = NoulAnswer(
                    question_id=q.id,
                    confidence=confidence,
                    result=bool(max_idx),
                )

        return PredictionResponse(answers=answers)
