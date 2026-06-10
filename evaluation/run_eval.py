"""
Run the adversarial rule evaluation.

Usage:
    python -m evaluation.run_eval                 # all rules
    python -m evaluation.run_eval spider_fusion   # one or more rules
"""
import sys

from zxdb.zxdb import ZXdb
from evaluation.harness import run_case, print_results
from evaluation.cases import RULES


def main(rule_names):
    zxdb = ZXdb()
    results = []
    try:
        for rule_name in rule_names:
            spec = RULES[rule_name]
            db_rule = getattr(zxdb, spec["db_method"])
            for case in spec["cases"]:
                case_name, builder = case[0], case[1]
                opts = case[2] if len(case) > 2 else {}
                results.append(
                    run_case(zxdb, rule_name, case_name, builder(),
                             db_rule, spec["pyzx"], **opts))
    finally:
        zxdb.close()
    all_ok = print_results(results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    names = sys.argv[1:] or list(RULES.keys())
    unknown = [n for n in names if n not in RULES]
    if unknown:
        print(f"Unknown rules: {unknown}. Available: {list(RULES.keys())}")
        sys.exit(2)
    main(names)
