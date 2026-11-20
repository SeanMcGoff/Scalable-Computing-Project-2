"""Claude used for QoL tooling around project"""

import json
import sys

from scripts.paths import GRADER_DIR, STUDENT_DIR


def main() -> None:
    """Run each graded entry point twice with the same seed and assert the
    outputs are identical (the autograder's seeded-determinism check)."""
    sys.path[:0] = [str(STUDENT_DIR), str(GRADER_DIR)]
    from policy import make_policy
    from simulator import run_one_hop, run_two_hop, run_variant_scenario

    params = json.loads((GRADER_DIR / "variant_params.json").read_text())
    grading = params["grading"]
    sim_time_ms = grading["sim_time_ms"]
    seeds = [grading["seed"], *grading["held_out_seeds"]]

    runners = {
        "run_one_hop": lambda s: run_one_hop(params, sim_time_ms, s),
        "run_two_hop": lambda s: run_two_hop(params, sim_time_ms, s),
        # fresh policy object per run so policy state cannot leak between runs
        "run_variant_scenario": lambda s: run_variant_scenario(
            params, sim_time_ms, s, make_policy(params)
        ),
    }

    failures = 0
    for name, run in runners.items():
        for seed in seeds:
            first, second = run(seed), run(seed)
            ok = first == second
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {name} seed={seed}: {first}")
            if not ok:
                print(f"     second run: {second}")

    if failures:
        sys.exit(f"{failures} determinism check(s) failed")
    print("All determinism checks passed")
