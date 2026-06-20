"""Extended golden-examples benchmark for physics skill acquisition.

Combines the original 5-question golden battery (golden-examples.txt) with the
15 CANDIDATE questions mined in skill-acquisition-physics-gemma-search.py into
one 20-question set, runs it against gemma_demo under three conditions
(base / in_context / adapter), and records, per question and condition:
    - pass/fail (numeric match against expected answer)
    - number of input tokens (context size: bare question vs question+DOC)
    - wall-clock time to compute (generation time; in_context also pays for
      a much longer prefill, adapter pays a one-time internalize() cost)

Produces matplotlib figures:
    accuracy_by_condition.png   - pass-rate bar chart, base vs in_context vs adapter
    context_tokens.png          - input token count per condition (log scale)
    time_to_compute.png         - per-question generation time, all conditions
                                   plus the one-time adapter internalize cost
    summary_dashboard.png       - all three combined into one figure

Run:
    /home/ubuntu/doc-to-lora/.venv/bin/python skill-acquisition-physics-benchmark.py
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

DOC = """\
# Newton's Laws and Classical Mechanics: Formula Reference

## Newton's First Law (Law of Inertia)
An object at rest stays at rest, and an object in motion stays in motion at
constant velocity, unless acted on by a net external force. If the net force
on an object is zero, its acceleration is zero.

## Newton's Second Law
The net force on an object equals its mass times its acceleration:
F = m * a
Equivalently: a = F / m, and m = F / a.
Units: F in newtons (N), m in kilograms (kg), a in meters per second squared
(m/s^2). 1 N = 1 kg*m/s^2.

## Newton's Third Law
For every action there is an equal and opposite reaction: the force object A
exerts on object B is equal in magnitude and opposite in direction to the
force object B exerts on object A.

## Weight
Weight is the force of gravity on an object near a planet's surface:
W = m * g
where g = 9.8 m/s^2 near Earth's surface.

## Friction
The force of friction between two surfaces:
F_friction = mu * N
where mu is the (dimensionless) coefficient of friction and N is the normal
force (on a flat horizontal surface, N = m * g).

## Momentum
Momentum is the product of mass and velocity:
p = m * v

## Impulse
Impulse equals the change in momentum, and also equals force times the time
interval over which it acts:
J = F * delta_t = delta_p

## Newton's Law of Universal Gravitation
The gravitational force between two masses:
F = G * m1 * m2 / r^2
where G = 6.674e-11 N*m^2/kg^2, m1 and m2 are the two masses, and r is the
distance between their centers.

## Centripetal Force
The net force required to keep an object of mass m moving in a circle of
radius r at speed v:
F_c = m * v^2 / r

## Kinematics (often combined with F = m * a)
v = v0 + a*t
x = x0 + v0*t + (1/2)*a*t^2
v^2 = v0^2 + 2*a*(x - x0)

# Worked Examples

Problem: A 4 kg object experiences a net force of 20 N. What is its
acceleration?
Solution: Let's think step by step. Newton's second law states F = m*a, so
a = F/m. Substituting, a = 20 N / 4 kg = 5 m/s^2.
Answer: 5 m/s^2

Problem: What is the weight of a 12 kg object on Earth's surface?
Solution: Let's think step by step. Weight is W = m*g. Using g = 9.8 m/s^2,
W = 12 kg * 9.8 m/s^2 = 117.6 N.
Answer: 117.6 N

Problem: A 20 kg crate rests on a horizontal floor with a coefficient of
friction of 0.25. What is the maximum friction force?
Solution: Let's think step by step. The normal force on a horizontal surface
equals the weight, N = m*g = 20 kg * 9.8 m/s^2 = 196 N. Friction is
F = mu*N = 0.25 * 196 N = 49 N.
Answer: 49 N

Problem: A 3 kg ball moves at 7 m/s. What is its momentum?
Solution: Let's think step by step. Momentum is p = m*v = 3 kg * 7 m/s =
21 kg*m/s.
Answer: 21 kg*m/s

Problem: A 0.4 kg ball is hit with a force of 15 N for 0.1 s. What is the
impulse (change in momentum)?
Solution: Let's think step by step. Impulse equals force times time,
J = F*delta_t = 15 N * 0.1 s = 1.5 kg*m/s. This equals the change in
momentum.
Answer: 1.5 kg*m/s

Problem: Two masses of 5 kg and 8 kg are 4 m apart. What is the
gravitational force between them?
Solution: Let's think step by step. Newton's law of gravitation gives
F = G*m1*m2/r^2. Substituting G = 6.674e-11 N*m^2/kg^2, m1 = 5 kg, m2 = 8 kg,
r = 4 m: F = 6.674e-11 * 5 * 8 / 16 = 6.674e-11 * 2.5 = 1.6685e-10 N.
Answer: 1.6685e-10 N

Problem: A 1.5 kg mass moves in a circle of radius 2 m at a speed of 6 m/s.
What centripetal force is required?
Solution: Let's think step by step. Centripetal force is
F_c = m*v^2/r = 1.5 kg * (6 m/s)^2 / 2 m = 1.5 * 36 / 2 = 27 N.
Answer: 27 N

Problem: A 500 kg car accelerates from rest to 15 m/s in 3 seconds. What net
force acted on the car?
Solution: Let's think step by step. First find acceleration using
kinematics: a = (v - v0)/t = (15 m/s - 0 m/s)/3 s = 5 m/s^2. Then apply
Newton's second law: F = m*a = 500 kg * 5 m/s^2 = 2500 N.
Answer: 2500 N

Problem: A 60 kg astronaut pushes on a 900 kg space module with a force of
40 N. What force does the module exert on the astronaut?
Solution: Let's think step by step. By Newton's third law, the force the
module exerts on the astronaut is equal in magnitude and opposite in
direction to the force the astronaut exerts on the module. Since the
astronaut pushes with 40 N, the module pushes back with 40 N.
Answer: 40 N

Problem: A 10 kg object has two horizontal forces acting on it: 25 N to the
right and 25 N to the left. What is its acceleration?
Solution: Let's think step by step. The net force is the vector sum:
25 N - 25 N = 0 N. By Newton's second law, a = F_net/m = 0 N / 10 kg =
0 m/s^2.
Answer: 0 m/s^2
"""

# Original 5 golden examples (golden-examples.txt) -- vetted against
# Qwen3-4B-Instruct-2507 -- plus the 15 CANDIDATE questions mined for
# gemma-2b in skill-acquisition-physics-gemma-search.py. Distinct numbers per
# question, covering every formula in DOC at least twice.
QUESTIONS = [
    dict(
        question="A 15 kg box sits on a horizontal floor with a coefficient of friction of 0.3. What is the maximum friction force?",
        expected=0.3 * (15 * 9.8),
        formula="F_friction = mu*N, N = m*g",
    ),
    dict(
        question="Two masses of 10 kg and 3 kg are 2 m apart. What is the gravitational force between them (G = 6.674e-11 N*m^2/kg^2)?",
        expected=6.674e-11 * 10 * 3 / 2**2,
        formula="F = G*m1*m2/r^2",
    ),
    dict(
        question="A 1000 kg car accelerates from rest to 20 m/s in 4 seconds. What net force acted on the car?",
        expected=1000 * (20 / 4),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question="A 1200 kg car traveling at 24 m/s skids to a stop on a horizontal road with a coefficient of friction of 0.4. What is the magnitude of the deceleration caused by friction?",
        expected=0.4 * 9.8,
        formula="F_friction = mu*m*g, a = F/m (mass cancels)",
    ),
    dict(
        question="A 1500 kg car traveling at 25 m/s comes to a stop in 5 seconds due to braking. What is the magnitude of the net braking force?",
        expected=1500 * (25 / 5),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question="A 25 kg crate sits on a horizontal floor with a coefficient of friction of 0.35. What is the maximum friction force?",
        expected=0.35 * (25 * 9.8),
        formula="F_friction = mu*N, N = m*g",
    ),
    dict(
        question="Two masses of 6 kg and 9 kg are 3 m apart. What is the gravitational force between them (G = 6.674e-11 N*m^2/kg^2)?",
        expected=6.674e-11 * 6 * 9 / 3**2,
        formula="F = G*m1*m2/r^2",
    ),
    dict(
        question="A 1800 kg car accelerates from rest to 18 m/s in 6 seconds. What net force acted on the car?",
        expected=1800 * (18 / 6),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question="A 900 kg car traveling at 20 m/s skids to a stop on a horizontal road with a coefficient of friction of 0.5. What is the magnitude of the deceleration caused by friction?",
        expected=0.5 * 9.8,
        formula="F_friction = mu*m*g, a = F/m (mass cancels)",
    ),
    dict(
        question="A 2000 kg truck traveling at 18 m/s comes to a stop in 6 seconds due to braking. What is the magnitude of the net braking force?",
        expected=2000 * (18 / 6),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question="What is the weight of a 25 kg object on Earth's surface (g = 9.8 m/s^2)?",
        expected=25 * 9.8,
        formula="W = m*g",
    ),
    dict(
        question="An 8 kg object moves at 12 m/s. What is its momentum?",
        expected=8 * 12,
        formula="p = m*v",
    ),
    dict(
        question="A 0.5 kg ball is hit with a force of 20 N for 0.3 s. What is the impulse (change in momentum)?",
        expected=20 * 0.3,
        formula="J = F*delta_t",
    ),
    dict(
        question="A 3 kg mass moves in a circle of radius 1.5 m at a speed of 9 m/s. What centripetal force is required?",
        expected=3 * 9**2 / 1.5,
        formula="F_c = m*v^2/r",
    ),
    dict(
        question="A 70 kg person pushes on a 1100 kg cart with a force of 55 N. What force does the cart exert on the person?",
        expected=55,
        formula="Newton's third law: equal and opposite",
    ),
    dict(
        question="A 5 kg block experiences two horizontal forces: 30 N to the right and 12 N to the left. What is its acceleration?",
        expected=(30 - 12) / 5,
        formula="F_net = sum of forces, a = F_net/m",
    ),
    dict(
        question="Two masses of 40 kg and 60 kg are 8 m apart. What is the gravitational force between them (G = 6.674e-11 N*m^2/kg^2)?",
        expected=6.674e-11 * 40 * 60 / 8**2,
        formula="F = G*m1*m2/r^2",
    ),
    dict(
        question="A 12 kg box on a horizontal floor has a coefficient of friction of 0.45. What is the maximum friction force?",
        expected=0.45 * (12 * 9.8),
        formula="F_friction = mu*N, N = m*g",
    ),
    dict(
        question="A 1400 kg car accelerates from 5 m/s to 25 m/s in 5 seconds. What net force acted on the car?",
        expected=1400 * ((25 - 5) / 5),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question="A 15 kg cart moving at 3 m/s is brought to a complete stop. What is the magnitude of the impulse delivered to it?",
        expected=15 * 3,
        formula="J = delta_p = m*delta_v",
    ),
]

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
    n_input_tokens = inputs["input_ids"].shape[1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = output_ids[0][n_input_tokens:]
    n_new_tokens = new_tokens.shape[0]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return (prefill + continuation).strip(), n_input_tokens, n_new_tokens, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2l-dir", default="trained_d2l/gemma_demo")
    parser.add_argument("--checkpoint", default="checkpoint-80000/pytorch_model.bin")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--assistant-prefill", default="Let's think step by step. ")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="physics_benchmark_results.json")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = f"{args.d2l_dir}/{args.checkpoint}"
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = ModulatedPretrainedModel.from_state_dict(state_dict, train=False, use_sequence_packing=False)
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)
    print(f"Loaded {args.d2l_dir}/{args.checkpoint} ({model.base_model.name_or_path})\n")

    # warmup -- discard first-call CUDA kernel JIT cost
    model.reset()
    generate_text(model, tokenizer, "Say hi.", 4, device)

    # Condition 1: base -- cold, no doc, no adapter.
    model.reset()
    base_results = [
        generate_text(model, tokenizer, q["question"], args.max_new_tokens, device, args.assistant_prefill)
        for q in QUESTIONS
    ]

    # Condition 2: in-context -- doc pasted into every prompt, still no adapter.
    model.reset()
    in_context_results = [
        generate_text(
            model, tokenizer, f"{DOC}\n\n{q['question']}", args.max_new_tokens, device, args.assistant_prefill
        )
        for q in QUESTIONS
    ]

    # Condition 3: D2L adapter -- internalize the doc ONCE, then bare questions.
    model.reset()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.internalize(DOC)
    if device.type == "cuda":
        torch.cuda.synchronize()
    internalize_time = time.perf_counter() - start
    adapter_results = [
        generate_text(model, tokenizer, q["question"], args.max_new_tokens, device, args.assistant_prefill)
        for q in QUESTIONS
    ]
    model.reset()
    print(f"(internalize: {internalize_time:.2f}s, one-time)\n")

    rows = []
    for q, base, ic, ad in zip(QUESTIONS, base_results, in_context_results, adapter_results):
        base_answer, base_in_tok, base_out_tok, base_time = base
        ic_answer, ic_in_tok, ic_out_tok, ic_time = ic
        ad_answer, ad_in_tok, ad_out_tok, ad_time = ad

        base_pass = check_numeric(base_answer, q["expected"])
        ic_pass = check_numeric(ic_answer, q["expected"])
        ad_pass = check_numeric(ad_answer, q["expected"])

        rows.append(dict(
            question=q["question"], formula=q["formula"], expected=q["expected"],
            base_answer=base_answer, base_pass=base_pass,
            base_input_tokens=base_in_tok, base_output_tokens=base_out_tok, base_time=base_time,
            in_context_answer=ic_answer, in_context_pass=ic_pass,
            in_context_input_tokens=ic_in_tok, in_context_output_tokens=ic_out_tok, in_context_time=ic_time,
            adapter_answer=ad_answer, adapter_pass=ad_pass,
            adapter_input_tokens=ad_in_tok, adapter_output_tokens=ad_out_tok, adapter_time=ad_time,
        ))
        print(f"{'=' * 80}\n{q['question']}  [{q['formula']}]\nExpected: {q['expected']:g}\n{'=' * 80}")
        print(f"--- base      ({'PASS' if base_pass else 'FAIL'}, {base_time:.2f}s, in={base_in_tok}) --- {base_answer[:120]}")
        print(f"--- in_context({'PASS' if ic_pass else 'FAIL'}, {ic_time:.2f}s, in={ic_in_tok}) --- {ic_answer[:120]}")
        print(f"--- adapter   ({'PASS' if ad_pass else 'FAIL'}, {ad_time:.2f}s, in={ad_in_tok}) --- {ad_answer[:120]}\n")

    n = len(rows)
    print(f"\n{'=' * 60}\nSUMMARY ({n} questions)\n{'=' * 60}")
    print(f"base={sum(r['base_pass'] for r in rows)}/{n}  "
          f"in_context={sum(r['in_context_pass'] for r in rows)}/{n}  "
          f"adapter={sum(r['adapter_pass'] for r in rows)}/{n}")

    output = dict(doc=DOC, internalize_time=internalize_time, rows=rows)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full transcript to {args.output_json}")

    make_plots(output, args.output_dir)


def make_plots(output, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = output["rows"]
    internalize_time = output["internalize_time"]
    n = len(rows)
    conditions = ["base", "in_context", "adapter"]
    labels = ["Base\n(cold)", "In-Context\n(doc in prompt)", "Adapter\n(D2L internalized)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    pass_rates = [100 * sum(r[f"{c}_pass"] for r in rows) / n for c in conditions]
    avg_input_tokens = [np.mean([r[f"{c}_input_tokens"] for r in rows]) for c in conditions]
    avg_time = [np.mean([r[f"{c}_time"] for r in rows]) for c in conditions]
    per_q_time = {c: [r[f"{c}_time"] for r in rows] for c in conditions}

    # 1. Accuracy bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, pass_rates, color=colors)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title(f"Physics word-problem accuracy ({n} questions)\ngemma-2b, greedy decoding")
    ax.set_ylim(0, 105)
    for bar, rate, c in zip(bars, pass_rates, conditions):
        passed = sum(r[f"{c}_pass"] for r in rows)
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{rate:.0f}%\n({passed}/{n})",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/accuracy_by_condition.png", dpi=150)
    plt.close(fig)

    # 2. Context tokens (log scale, since DOC dwarfs bare questions)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, avg_input_tokens, color=colors)
    ax.set_ylabel("Avg. input (context) tokens per question")
    ax.set_yscale("log")
    ax.set_title("Context size consumed per question")
    for bar, val in zip(bars, avg_input_tokens):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.0f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/context_tokens.png", dpi=150)
    plt.close(fig)

    # 3. Time to compute: per-question generation time (box/strip) + one-time internalize cost
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(len(conditions)))
    parts = ax.violinplot([per_q_time[c] for c in conditions], positions=positions, showmeans=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    for i, c in enumerate(conditions):
        ys = per_q_time[c]
        ax.scatter([i] * len(ys), ys, color="black", s=10, alpha=0.5, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Generation time per question (s)")
    ax.set_title("Time to compute per question\n(adapter also pays a one-time internalize cost, shown separately)")
    ax.annotate(f"one-time internalize: {internalize_time:.2f}s", xy=(2, max(per_q_time["adapter"])),
                xytext=(0.05, 0.92), textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    fig.savefig(f"{output_dir}/time_to_compute.png", dpi=150)
    plt.close(fig)

    # 4. Combined dashboard
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    bars = ax.bar(labels, pass_rates, color=colors)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 105)
    for bar, rate, c in zip(bars, pass_rates, conditions):
        passed = sum(r[f"{c}_pass"] for r in rows)
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{rate:.0f}%\n({passed}/{n})",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, avg_input_tokens, color=colors)
    ax.set_ylabel("Avg. input tokens")
    ax.set_yscale("log")
    ax.set_title("Context size per question")
    for bar, val in zip(bars, avg_input_tokens):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.0f}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    bars = ax.bar(labels, avg_time, color=colors)
    ax.set_ylabel("Avg. generation time (s)")
    ax.set_title("Time to compute per question")
    for bar, val in zip(bars, avg_time):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}s",
                ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Physics skill acquisition: base vs. in-context vs. D2L adapter ({n} questions, gemma-2b)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{output_dir}/summary_dashboard.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved plots to {output_dir}/accuracy_by_condition.png, context_tokens.png, "
          f"time_to_compute.png, summary_dashboard.png")


if __name__ == "__main__":
    main()
