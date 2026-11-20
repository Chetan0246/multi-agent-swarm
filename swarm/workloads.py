"""Synthetic workloads with exact, machine-checkable answers."""
from __future__ import annotations

import random

from .task import Task

WORDS = ["swarm", "orchestrator", "latency", "throughput", "token", "agent",
         "pipeline", "validator", "escalation", "concurrency", "batching", "cache"]


def mixed_workload(n: int, seed: int = 1, dup_frac: float = 0.05) -> list[Task]:
    """Arithmetic, string and counting tasks across 3 difficulty levels."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n):
        kind = rng.random()
        if kind < 0.5:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            t = Task(f"t{i}", f"What is {a} + {b}? Answer with the number only.", 1, expected=str(a + b))
        elif kind < 0.8:
            w = rng.choice(WORDS)
            t = Task(f"t{i}", f"Reverse the string '{w}'. Answer with the reversed string only.",
                     2, expected=w[::-1])
        else:
            w = rng.choice(WORDS) + rng.choice(WORDS)
            ch = rng.choice(sorted(set(w)))
            t = Task(f"t{i}", f"How many times does the letter '{ch}' appear in '{w}'? "
                              f"Answer with the number only.", 3, expected=str(w.count(ch)))
        tasks.append(t)
    # Real workloads repeat themselves; dedupe should catch it.
    for j in range(int(n * dup_frac)):
        src = tasks[rng.randrange(len(tasks))]
        tasks[rng.randrange(len(tasks))] = Task(f"d{j}", src.prompt, src.difficulty, expected=src.expected)
    # Keep ids unique after overwrites.
    seen: set[str] = set()
    out = []
    for k, t in enumerate(tasks):
        if t.id in seen:
            t.id = f"{t.id}_{k}"
        seen.add(t.id)
        out.append(t)
    return out
