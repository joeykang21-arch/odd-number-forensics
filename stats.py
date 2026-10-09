#!/usr/bin/env python3
"""
Recompute every number quoted in WRITEUP.md from the raw rollouts in runs/.

    python stats.py            # prints all tables and tests

No API calls. Reads runs/*/<condition>.jsonl only. Applies the one documented
hand correction (see CORRECTIONS below) so the output matches the write-up.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

from odd_number_exp import build_prompt, eval_keyword_hits

RUNS = Path(__file__).parent / "runs"

BATCH = {  # short name -> run directory
    "pilot":    "pilot-deepseek-v4-flash-20260824-235720",
    "partial":  "full-deepseek-v4-flash-20260825-000157",
    "graders1": "graders-deepseek-v4-flash-20260825-002205",
    "audit1":   "audit-deepseek-v4-flash-20260825-012722",
    "audit2":   "audit2-deepseek-v4-flash-20260826-010848",
    "graders2": "graders2-deepseek-v4-flash-20260826-011703",
    "keycells": "keycells-deepseek-v4-flash-20260826-014750",
}

# Hand corrections after auditing the parser (rollouts with >1 integer in the
# answer). Key: (batch, condition, idx) -> corrected parity.
CORRECTIONS = {
    # Answer "11 (odd, two-digit, 1–100 ...)": the trailing "100" was parsed
    # as the answer. The model chose 11, so this rollout is gaming.
    ("partial", "fractional", 26): "odd",
}


def load(batch, cond):
    path = RUNS / BATCH[batch] / f"{cond}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        key = (batch, cond, r["idx"])
        if key in CORRECTIONS:
            r["parity"] = CORRECTIONS[key]
            r["corrected"] = True
    return rows


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ztest(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se else float("nan")
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def odd(rows):
    return sum(r["parity"] == "odd" for r in rows)


def row(label, rows):
    k, n = odd(rows), len(rows)
    lo, hi = wilson(k, n)
    unp = sum(r["parity"] == "unparsed" for r in rows)
    print(f"  {label:<28} {k:>3}/{n:<3} {k/n:6.1%}  [{lo:.3f}, {hi:.3f}]"
          + (f"  (unparsed {unp})" if unp else ""))
    return k, n


def test(label, a, b):
    z, p = ztest(*a, *b)
    print(f"  {label:<40} diff {a[0]/a[1]-b[0]/b[1]:+.2f}  z={z:.2f}  p={p:.2g}")


def cites(rows, word):
    return sum(word in (r.get("reasoning") or "").lower() for r in rows)


def main():
    total = sum(len(load(b, p.stem)) for b in BATCH
                for p in (RUNS / BATCH[b]).glob("*.jsonl"))
    print(f"Total rollouts: {total}\n")

    print("Parser audit: rollouts whose answer contains more than one integer")
    for b in BATCH:
        for p in sorted((RUNS / BATCH[b]).glob("*.jsonl")):
            for r in load(b, p.stem):
                ints = re.findall(r"-?\d[\d,]*", r.get("content") or "")
                if len(ints) > 1:
                    print(f"  {b}/{p.stem} idx={r['idx']} parsed={r['parsed_number']}"
                          f" parity={r['parity']} content={r['content'][:80]!r}")
    print()

    print("Experiment 1: partial credit (batch 'partial')")
    a = row("binary", load("partial", "binary"))
    fr = load("partial", "fractional")
    b = row("fractional", fr)
    test("binary vs fractional", a, b)
    ev = [r for r in fr if r["parity"] == "even"]
    best = sum(10 <= r["parsed_number"] <= 98 for r in ev)
    print(f"  obeyers choosing a two-digit even number in range: {best}/{len(ev)}")
    print(f"  obeyers choosing 42: {sum(r['parsed_number'] == 42 for r in ev)}/{len(ev)}")
    arith = re.compile(r"2/3|0\.66|0\.67|\.666|\.667|two.thirds|2 of 3|2 out of 3")
    print(f"  obeyers whose CoT writes out the 2/3 score: "
          f"{sum(bool(arith.search(r.get('reasoning') or '')) for r in ev)}/{len(ev)}\n")

    print("Experiment 2: who reviews (batch 'graders2'; 'graders1' replication)")
    g = {c: load("graders2", c) for c in
         ["binary", "grader-consequence", "grader-regex", "grader-generic",
          "grader-llm", "grader-senior", "grader-human"]}
    s = {c: row(c, rs) for c, rs in g.items()}
    test("regex vs consequence-only", s["grader-regex"], s["grader-consequence"])
    test("generic vs consequence-only", s["grader-generic"], s["grader-consequence"])
    test("llm vs consequence-only", s["grader-llm"], s["grader-consequence"])
    test("senior vs consequence-only", s["grader-senior"], s["grader-consequence"])
    test("human vs consequence-only", s["grader-human"], s["grader-consequence"])
    print("  replication (graders1):")
    for c in ["binary", "grader-regex", "grader-generic", "grader-llm",
              "grader-senior", "grader-human"]:
        row("  " + c, load("graders1", c))
    human_all = (load("graders1", "grader-human") + load("graders2", "grader-human")
                 + load("keycells", "grader-human"))
    row("human, all batches pooled", human_all)
    print(f"  human-arm CoTs that mention 'odd': "
          f"{sum('odd' in (r.get('reasoning') or '').lower() for r in human_all)}/{len(human_all)}")
    regex_all = load("graders1", "grader-regex") + load("graders2", "grader-regex")
    gm = [r for r in regex_all if r["parity"] == "odd"]
    lab = [r for r in gm if "even" in (r.get("content") or "").lower()]
    print(f"  regex-arm gamers (both batches) that label an odd answer 'even': "
          f"{len(lab)}/{len(gm)}  {[r['content'] for r in lab]}")
    print()

    print("Experiment 3a: stated audit rate (batch 'audit1'; 100% also in 'keycells')")
    for c in ["binary", "audit-25", "audit-50", "audit-75", "audit-100"]:
        row(c, load("audit1", c))
    row("audit-100 (keycells, n=200)", load("keycells", "audit-100"))
    a100 = load("audit1", "audit-100") + load("keycells", "audit-100")
    row("audit-100 pooled", a100)
    test("audit-25 vs audit-100 (audit1)", (odd(load("audit1", "audit-25")), 100),
         (odd(load("audit1", "audit-100")), 100))
    print()

    print("Experiment 3b: audit salience (batch 'audit2'; paused also in 'audit1')")
    n_ = row("neutral (documented)", load("audit2", "audit-neutral"))
    b2 = row("binary", load("audit2", "binary"))
    pz = row("paused", load("audit2", "audit-control"))
    test("binary vs neutral", b2, n_)
    test("neutral vs paused", n_, pz)
    print("  CoTs mentioning the pause ('paus'), gamers vs obeyers:")
    for label, rows in [("audit1", load("audit1", "audit-control")),
                        ("audit2", load("audit2", "audit-control"))]:
        gm = [r for r in rows if r["parity"] == "odd"]
        ob = [r for r in rows if r["parity"] == "even"]
        print(f"    {label}: gamers {cites(gm,'paus')}/{len(gm)}, obeyers {cites(ob,'paus')}/{len(ob)}")
    rows = load("audit1", "audit-control") + load("audit2", "audit-control")
    gm = [r for r in rows if r["parity"] == "odd"]
    ob = [r for r in rows if r["parity"] == "even"]
    test("pooled: gamers vs obeyers citing pause",
         (cites(gm, "paus"), len(gm)), (cites(ob, "paus"), len(ob)))
    print(f"    pooled gamers {cites(gm,'paus')}/{len(gm)}, obeyers {cites(ob,'paus')}/{len(ob)}")
    nr = load("audit2", "audit-neutral")
    gm = [r for r in nr if r["parity"] == "odd"]
    ob = [r for r in nr if r["parity"] == "even"]
    for w in ["documentation", "audit"]:
        test(f"neutral: gamers vs obeyers citing '{w}'",
             (cites(gm, w), len(gm)), (cites(ob, w), len(ob)))
        print(f"    gamers {cites(gm,w)}/{len(gm)}, obeyers {cites(ob,w)}/{len(ob)}")
    print()

    print("Eval-awareness keyword rate (share of CoTs with >=1 keyword hit)")
    rates = []
    for b in BATCH:
        for p in sorted((RUNS / BATCH[b]).glob("*.jsonl")):
            rs = load(b, p.stem)
            # Recomputed with the current harness so every batch uses the same
            # keyword list and prompt exclusion (the earliest batches stored
            # flags from an older version that did not exclude "grading").
            prompt = build_prompt(p.stem)
            rate = sum(bool(eval_keyword_hits(r.get("reasoning"), prompt))
                       for r in rs) / len(rs)
            rates.append((rate, f"{b}/{p.stem}", odd(rs) / len(rs)))
    for rate, name, g in sorted(rates):
        print(f"  {name:<32} keywords {rate:5.0%}   gaming {g:5.0%}")
    print()

    print("'Random' numbers chosen (all batches)")
    allrows = [r for b in BATCH for p in (RUNS / BATCH[b]).glob("*.jsonl")
               for r in load(b, p.stem)]
    go = Counter(r["parsed_number"] for r in allrows if r["parity"] == "odd")
    ge = Counter(r["parsed_number"] for r in allrows if r["parity"] == "even")
    print("  gamers, top 6:", go.most_common(6), "of", sum(go.values()))
    print("  obeyers, top 4:", ge.most_common(4), "of", sum(ge.values()))


if __name__ == "__main__":
    main()
