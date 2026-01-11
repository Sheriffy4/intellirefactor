#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXCLUDE_ACTIONABLE = {
    "external",
    "star_import",
    "dynamic_attribute",
    "baseclass_method",
    "class_instantiation",
}
EXCLUDE_INTERNAL = {"external", "star_import"}

CAPITALIZED_RE = re.compile(r"^[A-Z]\w*$")

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_call_graph(data: dict):
    # поддержка разных форматов отчётов
    cg = data.get("call_graph", {})
    edges = cg.get("edges") or data.get("call_edges") or []
    unresolved = cg.get("unresolved_calls") or data.get("unresolved_calls") or []
    return edges, unresolved

def short_file(path: str) -> str:
    # компактнее для вывода
    try:
        p = Path(path)
        return str(p).replace("\\", "/")
    except Exception:
        return path

def print_top(counter: Counter, title: str, n: int):
    print(f"\n== {title} (top {n}) ==")
    for k, v in counter.most_common(n):
        print(f"{v:>6}  {k}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="path to functional_map.json")
    ap.add_argument("--top", type=int, default=30, help="top N for listings")
    ap.add_argument("--dump-examples", type=int, default=3, help="examples per pattern")
    args = ap.parse_args()

    data = load_json(args.json_path)
    edges, unresolved = get_call_graph(data)

    # counters
    by_reason = Counter(u.get("reason", "not_found") for u in unresolved)

    internal_unresolved = [u for u in unresolved if u.get("reason") not in EXCLUDE_INTERNAL]
    actionable_unresolved = [u for u in unresolved if u.get("reason") not in EXCLUDE_ACTIONABLE]

    total_calls = len(edges) + len(unresolved)
    internal_total = len(edges) + len(internal_unresolved)
    actionable_total = len(edges) + len(actionable_unresolved)

    overall_rate = (len(edges) / total_calls) if total_calls else 0.0
    internal_rate = (len(edges) / internal_total) if internal_total else 0.0
    actionable_rate = (len(edges) / actionable_total) if actionable_total else 0.0

    print("== Totals ==")
    print(f"edges(resolved):          {len(edges)}")
    print(f"unresolved total:         {len(unresolved)}")
    print(f"total calls:              {total_calls}")
    print("")
    print(f"overall rate:             {overall_rate:.1%}")
    print(f"internal rate:            {internal_rate:.1%}")
    print(f"actionable rate:          {actionable_rate:.1%}")
    print("")
    print(f"internal unresolved:      {len(internal_unresolved)}")
    print(f"actionable unresolved:    {len(actionable_unresolved)}")

    print_top(by_reason, "Unresolved by reason", args.top)

    # actionable diagnostics
    ar_reason = Counter(u.get("reason", "not_found") for u in actionable_unresolved)
    ar_calls = Counter(u.get("raw_call", "") for u in actionable_unresolved)
    ar_files = Counter(short_file(u.get("file", "")) for u in actionable_unresolved)
    ar_callers = Counter(u.get("caller_id", "") for u in actionable_unresolved)

    print_top(ar_reason, "ACTIONABLE unresolved by reason", args.top)
    print_top(ar_calls, "ACTIONABLE raw_call", args.top)
    print_top(ar_files, "ACTIONABLE files", args.top)
    print_top(ar_callers, "ACTIONABLE callers", min(args.top, 15))

    # 1) constructor-like not_found (CamelCase) -> часто это классы (RefactoringPlan, RefactoringStep)
    ctor = [u for u in actionable_unresolved
            if u.get("reason") == "not_found" and CAPITALIZED_RE.match(u.get("raw_call", ""))]
    ctor_calls = Counter(u["raw_call"] for u in ctor)
    print_top(ctor_calls, "Constructor-like NOT_FOUND (CamelCase)", args.top)

    # примеры
    if ctor_calls:
        print("\nExamples for constructor-like calls:")
        ex = defaultdict(list)
        for u in ctor:
            key = u["raw_call"]
            if len(ex[key]) < args.dump_examples:
                ex[key].append(f'{short_file(u.get("file",""))}:{u.get("lineno")}  caller={u.get("caller_id")}')
        for name, _ in ctor_calls.most_common(min(args.top, 10)):
            for line in ex[name]:
                print(f"  {name}: {line}")

    non_ctor_nf = [u for u in actionable_unresolved
                  if u.get("reason") == "not_found"
                  and not CAPITALIZED_RE.match(u.get("raw_call",""))]
    print_top(Counter(u["raw_call"] for u in non_ctor_nf), "NOT_FOUND without CamelCase (real leftovers)", args.top)
    
    # 2) attribute-head histogram for actionable (помогает понять: это alias? модуль? объект?)
    attr = [u for u in actionable_unresolved if "." in (u.get("raw_call") or "")]
    attr_head = Counter((u["raw_call"].split(".", 1)[0]) for u in attr)
    print_top(attr_head, "ACTIONABLE attribute-call heads", args.top)

if __name__ == "__main__":
    main()