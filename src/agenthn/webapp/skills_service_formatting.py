"""Service behind the live self-improving skills demo (formatting track).

Same base -> in-context -> D2L-adapter comparison as skills_service_physics.py,
but the skill is output formatting (JSON / YAML / protobuf / bulleted lists,
see ``skills/formatting_bench.py``) instead of physics word problems: the
base model is asked cold, then with the format reference doc pasted into
every prompt (in-context ceiling), then with the doc internalized into a
LoRA adapter ONCE and nothing but the bare task in the prompt. Grading checks
structural validity for the task's `kind`, not a numeric answer.

Runs on the shared, lock-protected D2L model (see runtime.py) -- only one
demo (memory / personalization / skills) can generate at a time.
"""

from __future__ import annotations

import time
from typing import Iterator

from ..skills.formatting_bench import DOC, QUESTIONS, is_correct
from .runtime import MODEL_LOCK, get_model
from .skills_common import generate_with_prefill

# No "let's think step by step" prefill here -- it doesn't make sense before
# emitting raw JSON/YAML/proto/bullets.
PREFILL = ""


class SkillsFormattingService:
    def __init__(self) -> None:
        self._lock = MODEL_LOCK

    def meta(self) -> dict:
        return {
            "questions": [
                {"question": q["question"], "formula": q["formula"], "expected": q["expected"]}
                for q in QUESTIONS
            ],
        }

    def run(self) -> Iterator[dict]:
        """Yield frames for the whole run (held under the model lock)."""
        with self._lock:
            model = get_model()
            doc_tokens = model.count_tokens(DOC)

            yield {
                "type": "meta",
                "questions": [
                    {"question": q["question"], "formula": q["formula"], "expected": q["expected"]}
                    for q in QUESTIONS
                ],
                "doc_tokens": doc_tokens,
            }

            tallies: dict[str, dict] = {}

            def run_phase(phase: str, prompt_for, prompt_tokens: int) -> Iterator[dict]:
                correct = 0
                total_time = 0.0
                for i, q in enumerate(QUESTIONS):
                    answer, elapsed = generate_with_prefill(model, prompt_for(q), prefill=PREFILL)
                    ok = is_correct(answer, q["kind"])
                    correct += ok
                    total_time += elapsed
                    yield {
                        "type": "answer",
                        "phase": phase,
                        "index": i,
                        "question": q["question"],
                        "formula": q["formula"],
                        "expected": q["expected"],
                        "answer": answer,
                        "correct": ok,
                        "elapsed": round(elapsed, 2),
                        "prompt_tokens": prompt_tokens,
                    }
                tallies[phase] = {
                    "correct": correct,
                    "total": len(QUESTIONS),
                    "total_time": round(total_time, 2),
                }

            # Phase 1: base model -- no doc, no adapter.
            model.reset()
            yield from run_phase("base", lambda q: q["question"], 0)

            # Phase 2: in-context ceiling -- doc pasted into every prompt.
            yield from run_phase(
                "in_context", lambda q: f"{DOC}\n\n{q['question']}", doc_tokens
            )

            # Phase 3: D2L adapter -- internalize the doc ONCE, then bare questions.
            model.reset()
            start = time.perf_counter()
            model.internalize(DOC)
            internalize_time = time.perf_counter() - start
            yield {"type": "internalize", "elapsed": round(internalize_time, 2)}
            yield from run_phase("adapter", lambda q: q["question"], 0)
            model.reset()

            yield {
                "type": "done",
                "summary": tallies,
                "internalize_time": round(internalize_time, 2),
                "doc_tokens": doc_tokens,
            }


def build_skills_formatting_service() -> SkillsFormattingService:
    return SkillsFormattingService()


__all__ = ["SkillsFormattingService", "build_skills_formatting_service"]
