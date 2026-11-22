"""Simulation validator"""

import json

import numpy as np
from simulator import run_one_hop, run_two_hop

from scripts.paths import GRADER_DIR


def main() -> None:
    params = json.loads((GRADER_DIR / "variant_params.json").read_text())
    grading = params["grading"]
    tol = grading["tolerances"]
    exp = params["reference"]["expected"]
    seeds = [grading["seed"], *grading["held_out_seeds"]]

    scenarios = {
        "one_hop": (run_one_hop, exp["one_hop_rate"], params["physics"]["f_link"]),
        "two_hop": (run_two_hop, exp["two_hop_rate"], exp["two_hop_fidelity"]),
    }

    for name, (run, exp_rate, exp_fid) in scenarios.items():
        print(f"\n{name}  expected rate {exp_rate:.4f}/s, fidelity {exp_fid:.4f}")
        for seed in seeds:
            res = run(params, grading["sim_time_ms"], seed)
            rate_err = (res["rate"] - exp_rate) / exp_rate
            fid_err = (res["mean_fidelity"] - exp_fid) / exp_fid
            ok = (
                abs(rate_err) <= tol["rate_rel"]
                and abs(res["mean_fidelity"] - exp_fid) <= tol["fid_abs"]
            )
            print(
                f"{'OK' if ok else 'FAIL'}\t seed {seed}: "
                f"rate {np.round(res['rate'], 4)} ({rate_err:+.2%}), "
                f"fidelity {np.round(res['mean_fidelity'], 4)} ({fid_err:+.2%})"
            )
