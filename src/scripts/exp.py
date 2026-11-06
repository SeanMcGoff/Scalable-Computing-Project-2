'''Claude used for QoL tooling around project'''

import json
import sys

from scripts.paths import GRADER_DIR, STUDENT_DIR


def main() -> None:
    sys.path[:0] = [str(STUDENT_DIR), str(GRADER_DIR)]
    from experiments import run_sensitivity

    params = json.loads((GRADER_DIR / "variant_params.json").read_text())
    out = run_sensitivity(params, params["experiment"]["seeds"])
    print(json.dumps(out, indent=2))
