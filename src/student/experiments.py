"""
experiments.py — QIS-NET Project 2 sensitivity experiment (STUDENT FILE).

Implement run_sensitivity: one-at-a-time (OAT) sensitivity of the variant
scenario's delivered RATE to memory T2 over params['experiment']['t2_grid_ms'],
using >= params['experiment']['seeds_per_point'] seeds per grid point.

Output contract (autograded): write experiment_results.json containing
{
  "parameter": "t2_ms",
  "grid": [...],
  "seeds_per_point": int,
  "points": [
     {"t2_ms": x, "rate_mean": m, "rate_ci95": h, "n_seeds": k}, ...
  ]
}
CI: 95% t-interval half-width over the per-seed rates.
"""

import copy
import json
import math

import numpy as np
from policy import make_policy
from simulator import run_variant_scenario


def mean_ci95(values):
    """Return (mean, 95% CI half-width) using the t-distribution."""
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, 0.0
    s = float(np.std(values, ddof=1))
    # t_{0.975, df} for small df (table lookup, no scipy needed)
    t_table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }
    t = t_table.get(n - 1, 1.96)
    return m, t * s / math.sqrt(n)


def run_sensitivity(params, seeds) -> dict:
    """TODO: implement the OAT sweep described above.
    For each t2 in params['experiment']['t2_grid_ms']:
      - deep-copy params, set params_copy['physics']['t2_ms'] = t2
      - for each seed in seeds[:seeds_per_point]: run the variant scenario
        with your policy (make_policy(params_copy)) and record 'rate'
      - append {"t2_ms": ..., "rate_mean": ..., "rate_ci95": ..., "n_seeds": ...}
    Write the results dict to experiment_results.json and return it.
    """

    exp = params["experiment"]
    t2_grid = exp["t2_grid_ms"]
    seeds_per_point = exp["seeds_per_point"]
    sim_time_ms = params["grading"]["sim_time_ms"]

    points = []
    # For each t2 in t2_grid_ms
    for t2 in t2_grid:
        # Deep copy params
        p_copy = copy.deepcopy(params)
        # set t2
        p_copy["physics"]["t2_ms"] = t2
        variant_rates = []
        # for each seed in seeds_per_point, run variant scenario
        for seed in seeds[:seeds_per_point]:
            # use own policy, record rate
            variant_rates.append(
                run_variant_scenario(p_copy, sim_time_ms, seed, make_policy(p_copy))[
                    "rate"
                ]
            )
        mean, ci95 = mean_ci95(variant_rates)
        # append t2, mean, ci95, n_seeds
        points.append(
            {
                "t2_ms": t2,
                "rate_mean": mean,
                "rate_ci95": ci95,
                "n_seeds": len(variant_rates),
            }
        )

    # Write to dict, json, and return
    sensitivity_result = {
        "parameter": "t2_ms",
        "grid": t2_grid,
        "seeds_per_point": seeds_per_point,
        "points": points,
    }
    with open("experiment_results.json", "w") as f:
        json.dump(sensitivity_result, f)
    return sensitivity_result


# Used for autograding, ignore and use `uv run exp` instead
if __name__ == "__main__":
    import sys

    params = json.load(
        open(sys.argv[1] if len(sys.argv) > 1 else "variant_params.json")
    )
    exp = params["experiment"]
    seeds = exp["seeds"]
    out = run_sensitivity(params, seeds)
    print(json.dumps(out, indent=2))
