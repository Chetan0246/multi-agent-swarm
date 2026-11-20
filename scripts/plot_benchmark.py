"""Render docs/assets/*.svg from a `swarm bench --out` JSON file.

    python -m swarm bench --seeds 3 --out docs/assets/benchmark.json
    python scripts/plot_benchmark.py docs/assets/benchmark.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCEN = ["naive", "v1-static", "v2-1agent", "v2-cold", "v2-warm"]
COLORS = {"naive": "#9aa0a6", "v1-static": "#5f8dd3", "v2-1agent": "#b58cd9",
          "v2-cold": "#f2a65a", "v2-warm": "#2a9d8f"}
PANELS = [
    ("cost_per_task", "Cost per task (billed-equiv. tokens, log)", True),
    ("wall_time_s", "Wall time (s)", False),
    ("call_p99_s", "p99 call latency (s)", False),
    ("transient_errors", "Calls wasted on 5xx during incident", False),
]


def main(path: str) -> None:
    data = json.loads(Path(path).read_text())
    profiles = list(data["profiles"])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.2))
    width = 0.8 / len(SCEN)
    for ax, (key, title, log) in zip(axes, PANELS):
        for i, sc in enumerate(SCEN):
            means = [data["profiles"][p]["mean"][sc][key] for p in profiles]
            errs = [data["profiles"][p]["ci95"][sc][key] for p in profiles]
            xs = [j + (i - (len(SCEN) - 1) / 2) * width for j in range(len(profiles))]
            ax.bar(xs, means, width, yerr=errs, capsize=2, color=COLORS[sc], label=sc)
        ax.set_xticks(range(len(profiles)), profiles)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if log:
            ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Simulated endpoint, {data['n']} tasks x {data['seeds']} seeds (mean, 95% CI)", fontsize=11)
    fig.tight_layout()
    out = Path(path).with_suffix(".svg")
    fig.savefig(out)
    print(f"wrote {out}")
    pareto(data, Path(path).with_name("pareto.svg"))


def pareto(data: dict, out: Path) -> None:
    """Cost vs wall time per scenario: the accuracy-cost frontier view of agent evaluation."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig, axes = plt.subplots(1, len(data["profiles"]), figsize=(15, 4.2))
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax, (prof, d) in zip(axes, data["profiles"].items()):
        for sc in SCEN:
            m, c = d["mean"][sc], d["ci95"][sc]
            ax.errorbar(m["wall_time_s"], m["cost_per_task"], xerr=c["wall_time_s"], yerr=c["cost_per_task"],
                        fmt="o", ms=8, color=COLORS[sc], ecolor=COLORS[sc], elinewidth=1, capsize=2,
                        label=f"{sc} ({m['success_%']:.1f}% ok)", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("wall time (s)")
        ax.set_title(prof, fontsize=10)
        ax.grid(alpha=0.25, which="major")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("cost per task (billed-equiv. tokens)")
    fig.suptitle("Cost vs wall time (log-log, 95% CI) - lower-left is better", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/benchmark.json")
