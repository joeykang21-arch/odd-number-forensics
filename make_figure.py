#!/usr/bin/env python3
"""Draw figures/gaming_by_condition.png from the raw rollouts (no API calls)."""
from pathlib import Path

import matplotlib.pyplot as plt

from stats import load, odd, wilson

BLUE, GRAY = "#2a78d6", "#a3a29c"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# (panel title, [(label, batch, condition, is_baseline), ...]) - one batch per panel
PANELS = [
    ("1. Partial credit", [
        ("Binary score (baseline)", "partial", "binary", True),
        ("Fractional score", "partial", "fractional", False),
    ]),
    ("2. Who reviews the answer", [
        ("No review sentence (baseline)", "graders2", "binary", True),
        ("Review, no reviewer named", "graders2", "grader-consequence", False),
        ("A regex script", "graders2", "grader-regex", False),
        ("A grader", "graders2", "grader-generic", False),
        ("An LLM grader", "graders2", "grader-llm", False),
        ("A senior engineer", "graders2", "grader-senior", False),
        ("A human", "graders2", "grader-human", False),
    ]),
    ("3a. Stated audit rate", [
        ("No audit block (baseline)", "audit1", "binary", True),
        ("25% audited", "audit1", "audit-25", False),
        ("50% audited", "audit1", "audit-50", False),
        ("75% audited", "audit1", "audit-75", False),
        ("100% audited", "audit1", "audit-100", False),
    ]),
    ("3b. Audit mentioned, no rate", [
        ("No audit block (baseline)", "audit2", "binary", True),
        ("Audits are documented", "audit2", "audit-neutral", False),
        ("Auditing is paused", "audit2", "audit-control", False),
    ]),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(
    len(PANELS), 1, figsize=(8, 8.2), sharex=True,
    gridspec_kw={"height_ratios": [len(p[1]) for p in PANELS], "hspace": 0.55},
)
fig.patch.set_facecolor(SURFACE)

for ax, (title, items) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    for i, (label, batch, cond, base) in enumerate(items):
        rows = load(batch, cond)
        k, n = odd(rows), len(rows)
        lo, hi = wilson(k, n)
        p = k / n
        y = len(items) - 1 - i
        ax.barh(y, p, height=0.62, color=GRAY if base else BLUE, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, zorder=3)
        ax.text(hi + 0.015, y, f"{p:.0%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in reversed(items)], color=INK)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=4)
    ax.set_xlim(0, 1.08)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)

axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axes[-1].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
axes[-1].set_xlabel("Share of rollouts that answer with an odd number (gaming)", color=MUTED)
fig.suptitle("DeepSeek V4 Flash, Odd Number environment",
             x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.text(0.02, 0.935, "n = 100 per bar. Gray bar = that batch's own baseline. "
         "Lines = Wilson 95% CI.", color=MUTED, fontsize=9)

out = Path(__file__).parent / "figures" / "gaming_by_condition.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {out}")
