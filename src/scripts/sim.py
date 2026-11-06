'''Claude used for QoL tooling around project'''

import json
import sys

from scripts.paths import GRADER_DIR, STUDENT_DIR


def main() -> None:
    sys.path[:0] = [str(STUDENT_DIR), str(GRADER_DIR)]
    from policy import make_policy
    from simulator import run_one_hop, run_two_hop, run_variant_scenario

    params = json.loads((GRADER_DIR / "variant_params.json").read_text())
    grading = params["grading"]
    slo = params["slo"]
    seeds = [grading["seed"], *grading["held_out_seeds"]]
    print(f"SLO: {slo}")
    for seed in seeds:
        print(f"seed {seed}: one_hop {run_one_hop(params, grading['sim_time_ms'], seed)}")
        print(f"seed {seed}: two_hop {run_two_hop(params, grading['sim_time_ms'], seed)}")
    for seed in seeds:
        res = run_variant_scenario(
            params, grading["sim_time_ms"], seed, make_policy(params)
        )
        ok = (
            res["rate"] >= slo["min_rate"]
            and res["mean_fidelity"] >= slo["min_fidelity"]
            and res["p95_latency_ms"] <= slo["max_p95_latency_ms"]
        )
        print(f"seed {seed}: {'PASS' if ok else 'FAIL'} {res}")
