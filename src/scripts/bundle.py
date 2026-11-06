'''Claude used for QoL tooling around project'''

import zipfile

from scripts.paths import ROOT, STUDENT_DIR

SUBMISSION_FILES = ["simulator.py", "policy.py", "experiments.py"]


def main() -> None:
    out = ROOT / "submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SUBMISSION_FILES:
            z.write(STUDENT_DIR / name, arcname=name)
    print(f"Wrote {out}: {', '.join(SUBMISSION_FILES)}")
