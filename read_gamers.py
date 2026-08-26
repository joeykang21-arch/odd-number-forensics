import json, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("parity", choices=["odd", "even", "all"])
ap.add_argument("--grep", help="only rollouts whose reasoning contains this (case-insens.)")
ap.add_argument("--head", type=int, default=800, help="chars from start of CoT")
ap.add_argument("--tail", type=int, default=800, help="chars from end of CoT")
ap.add_argument("--count", action="store_true", help="just count matches, no text")
args = ap.parse_args()

hits = 0
for line in open(args.path, encoding="utf-8"):
    r = json.loads(line)
    if args.parity != "all" and r.get("parity") != args.parity:
        continue
    cot = r.get("reasoning") or ""
    if args.grep and args.grep.lower() not in cot.lower():
        continue
    hits += 1
    if args.count:
        continue
    print(f"=== idx {r['idx']}  n={r.get('parsed_number')}  parity={r.get('parity')}"
          f"  kw={r.get('eval_kw_hits')}  criteria={r.get('criteria')}")
    if len(cot) <= args.head + args.tail:
        print(cot)
    else:
        print(cot[:args.head])
        print(f"\n[... {len(cot)-args.head-args.tail} chars omitted ...]\n")
        print(cot[-args.tail:])
    print()
print(f"# matched: {hits}", file=sys.stderr)