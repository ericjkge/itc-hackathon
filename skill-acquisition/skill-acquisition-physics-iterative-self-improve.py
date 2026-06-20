"""Experiment: iterative self-improvement of the internalized DOC.

Instead of a fixed, hand-written DOC, the BASE model writes its own doc, given
a nudging prompt asking it to lay out the relevant Newton's-law formulas and
any other facts useful for a specific physics question. We then:

  1. Generate doc v1 with the base model (nudged for relevant formulas).
  2. internalize(doc v1) -> LoRA v1, ask the question, observe the answer.
  3. Ask the (still v1-adapted) model to critique its own answer and produce
     an IMPROVED doc (v2) that adds whatever formulas/facts it thinks are
     missing or would help.
  4. internalize(doc v2) -> LoRA v2, ask the question again, observe+critique.
  5. Repeat once more to get doc v3 / LoRA v3 and a final answer.

So per question we do 3 internalize+ask+critique turns. At the end we compare
the ORIGINAL base-model answer (no doc, no adapter at all) against the turn-3
adapter answer, on 5 of the golden-examples.txt physics problems.

Run:
    /home/ubuntu/doc-to-lora/.venv/bin/python scripts/skill-acquisition-physics-iterative-self-improve.py
"""

import argparse
import importlib.machinery
import json
import math
import re
import sys
import time
import types

_deepspeed_stub = types.ModuleType("deepspeed")
_deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
sys.modules.setdefault("deepspeed", _deepspeed_stub)

import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# First 5 questions from itc-hackathon/skill-acquisition/golden-examples.txt
QUESTIONS = [
    dict(
        question=(
            "A 15 kg box sits on a horizontal floor with a coefficient of "
            "friction of 0.3. What is the maximum friction force?"
        ),
        expected=0.3 * (15 * 9.8),
        formula="F_friction = mu*N, N = m*g",
    ),
    dict(
        question=(
            "Two masses of 10 kg and 3 kg are 2 m apart. What is the "
            "gravitational force between them (G = 6.674e-11 N*m^2/kg^2)?"
        ),
        expected=6.674e-11 * 10 * 3 / 2**2,
        formula="F = G*m1*m2/r^2",
    ),
    dict(
        question=(
            "A 1000 kg car accelerates from rest to 20 m/s in 4 seconds. "
            "What net force acted on the car?"
        ),
        expected=1000 * (20 / 4),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question=(
            "A 1200 kg car traveling at 24 m/s skids to a stop on a "
            "horizontal road with a coefficient of friction of 0.4. What is "
            "the magnitude of the deceleration caused by friction?"
        ),
        expected=0.4 * 9.8,
        formula="F_friction = mu*m*g, a = F/m (mass cancels)",
    ),
    dict(
        question=(
            "A 1500 kg car traveling at 25 m/s comes to a stop in 5 seconds "
            "due to braking. What is the magnitude of the net braking force?"
        ),
        expected=1500 * (25 / 5),
        formula="a = (v-v0)/t, F = m*a",
    ),
]

N_TURNS = 3

NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\*{0,2}\d+\.\s")


def normalize_scientific_notation(text):
    text = LIST_MARKER_RE.sub("", text)
    text = text.translate(SUPERSCRIPT_MAP)
    return re.sub(r"(\d+\.?\d*)\s*[×x\*]\s*10\^?(-?\d+)", r"\1e\2", text)


def check_numeric(answer, expected, rel_tol=0.02):
    for match in NUMBER_RE.finditer(normalize_scientific_notation(answer)):
        try:
            val = float(match.group())
        except ValueError:
            continue
        if math.isclose(val, expected, rel_tol=rel_tol):
            return True
    return False


def generate_text(model, tokenizer, user_message, max_new_tokens, device, prefill=""):
    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chat_str += prefill
    inputs = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return (prefill + continuation).strip()


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


# --- Prompts driving the self-improvement loop -----------------------------

DOC_GEN_PROMPT = """\
You are preparing a short study note to help solve the following physics \
problem:

{question}

Write a concise reference note (not the final numeric answer) listing every \
physics law, formula, and definition that is relevant or could be helpful \
for solving this problem. Be generous: include Newton's laws, the specific \
equations needed (e.g. F = m*a, F_friction = mu*N, kinematics equations, \
gravitation, momentum, etc.), the units involved, and any constants (like \
g = 9.8 m/s^2 or G = 6.674e-11 N*m^2/kg^2) that might be needed. Do not solve \
the problem -- just write the reference note of laws/formulas/facts."""

CRITIQUE_AND_REVISE_PROMPT = """\
Here is a physics problem:
{question}

Here is a reference note of formulas/facts you previously wrote for this \
problem:
--- NOTE (v{turn}) ---
{doc}
--- END NOTE ---

Here is the answer you produced for this problem using that note:
--- ANSWER ---
{answer}
--- END ANSWER ---

Critique your own answer: is it numerically correct, did it use the right \
formula, did it make an arithmetic or unit mistake, is anything missing? \
Then write an IMPROVED reference note (v{next_turn}) that adds whatever \
additional formulas, facts, definitions, units, or worked-out general \
relationships would help you avoid the mistake and solve this problem \
correctly next time. Keep everything from the old note that is still useful, \
and add to it -- do not just repeat the old note unchanged. Output ONLY the \
improved reference note text, not the critique itself."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2l-dir", default="trained_d2l/gemma_demo")
    parser.add_argument("--checkpoint", default="checkpoint-80000/pytorch_model.bin")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--doc-max-new-tokens", type=int, default=400)
    parser.add_argument("--assistant-prefill", default="Let's think step by step. ")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = f"{args.d2l_dir}/{args.checkpoint}"
    state_dict, load_time = timed(
        lambda: torch.load(checkpoint_path, weights_only=False, map_location=device), device
    )
    model = ModulatedPretrainedModel.from_state_dict(state_dict, train=False, use_sequence_packing=False)
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)
    print(f"Loaded {args.d2l_dir}/{args.checkpoint} ({model.base_model.name_or_path}) in {load_time:.2f}s\n")

    all_results = []

    for qi, q in enumerate(QUESTIONS, 1):
        print(f"\n{'#' * 90}\nQUESTION {qi}: {q['question']}  [{q['formula']}]\nExpected: {q['expected']:g}\n{'#' * 90}")

        # --- Turn 0: pure base model, no doc, no adapter at all -----------
        model.reset()
        base_answer, base_time = timed(
            lambda: generate_text(model, tokenizer, q["question"], args.max_new_tokens, device, args.assistant_prefill),
            device,
        )
        base_pass = check_numeric(base_answer, q["expected"])
        print(f"\n--- TURN 0: base model (no doc, no adapter) [{'PASS' if base_pass else 'FAIL'}] ---\n{base_answer}")

        turns = []
        doc = None
        for turn in range(1, N_TURNS + 1):
            # 1. Generate (or revise) the doc with the BASE model (reset --
            #    no adapter active while writing/critiquing the note itself).
            model.reset()
            if turn == 1:
                doc_prompt = DOC_GEN_PROMPT.format(question=q["question"])
                doc, doc_gen_time = timed(
                    lambda p=doc_prompt: generate_text(model, tokenizer, p, args.doc_max_new_tokens, device),
                    device,
                )
            else:
                revise_prompt = CRITIQUE_AND_REVISE_PROMPT.format(
                    question=q["question"], turn=turn - 1, doc=doc,
                    answer=turns[-1]["answer"], next_turn=turn,
                )
                doc, doc_gen_time = timed(
                    lambda p=revise_prompt: generate_text(model, tokenizer, p, args.doc_max_new_tokens, device),
                    device,
                )

            # 2. internalize the (possibly revised) doc -> LoRA for this turn.
            model.reset()
            _, internalize_time = timed(lambda: model.internalize(doc), device)

            # 3. Ask the bare question through the new adapter.
            answer, ask_time = timed(
                lambda: generate_text(model, tokenizer, q["question"], args.max_new_tokens, device, args.assistant_prefill),
                device,
            )
            ans_pass = check_numeric(answer, q["expected"])

            print(f"\n--- TURN {turn}: doc (gen {doc_gen_time:.1f}s) ---\n{doc}")
            print(f"\n--- TURN {turn}: adapter answer (internalize {internalize_time:.1f}s, "
                  f"ask {ask_time:.1f}s) [{'PASS' if ans_pass else 'FAIL'}] ---\n{answer}")

            turns.append(dict(
                turn=turn, doc=doc, answer=answer, pass_=ans_pass,
                doc_gen_time=doc_gen_time, internalize_time=internalize_time, ask_time=ask_time,
            ))

        model.reset()
        final_pass = turns[-1]["pass_"]
        improved = final_pass and not base_pass
        regressed = base_pass and not final_pass
        print(f"\n=== Q{qi} SUMMARY === base={'PASS' if base_pass else 'FAIL'} -> "
              f"turn{N_TURNS}={'PASS' if final_pass else 'FAIL'}  "
              f"({'IMPROVED' if improved else 'REGRESSED' if regressed else 'NO CHANGE'})")

        all_results.append(dict(
            index=qi, question=q["question"], formula=q["formula"], expected=q["expected"],
            base_answer=base_answer, base_pass=base_pass, base_time=base_time,
            turns=turns,
            final_pass=final_pass, improved=improved, regressed=regressed,
        ))

    n = len(all_results)
    base_passes = sum(r["base_pass"] for r in all_results)
    final_passes = sum(r["final_pass"] for r in all_results)
    n_improved = sum(r["improved"] for r in all_results)
    n_regressed = sum(r["regressed"] for r in all_results)

    print(f"\n\n{'=' * 70}\nFINAL RESULTS: base model vs turn-{N_TURNS} self-improved adapter "
          f"({n} questions)\n{'=' * 70}")
    print(f"{'#':<3} {'base':<6} {'turn'+str(N_TURNS):<7} outcome  question")
    for r in all_results:
        outcome = "IMPROVED" if r["improved"] else "REGRESSED" if r["regressed"] else "same"
        print(f"{r['index']:<3} {'PASS' if r['base_pass'] else 'FAIL':<6} "
              f"{'PASS' if r['final_pass'] else 'FAIL':<7} {outcome:<8} {r['question'][:60]}")
    print(f"\nbase:            {base_passes}/{n}")
    print(f"turn-{N_TURNS} adapter: {final_passes}/{n}")
    print(f"improved (FAIL->PASS): {n_improved}/{n}, regressed (PASS->FAIL): {n_regressed}/{n}")

    output_path = (
        args.output
        or f"{args.d2l_dir}/skill_acquisition_physics_iterative_self_improve_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump({"n_turns": N_TURNS, "results": all_results}, f, indent=2)
    print(f"\nSaved full transcript to {output_path}")


if __name__ == "__main__":
    main()
