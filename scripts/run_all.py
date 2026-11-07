"""Reproduce the whole project with one command.

    python -m scripts.run_all                 # everything (~2.5 h on an RTX 5060 laptop)
    python -m scripts.run_all --from detection   # resume from a step
"""
from __future__ import annotations

import argparse
import subprocess
import sys

STEPS = [
    ("prepare", ["scripts.prepare_data"]),
    ("classifier", ["scripts.train_classifier", "--epochs", "30"]),
    ("nocrop", ["scripts.train_classifier", "--kind", "full", "--epochs", "12", "--tag", "nocrop"]),
    ("clip", ["scripts.train_clip", "--epochs", "15"]),
    ("detector", ["scripts.train_detector", "--epochs", "15"]),
    ("detection", ["scripts.run_detection"]),
    ("classical", ["scripts.train_classical"]),
    ("evaluate", ["scripts.evaluate", "--skip_robust"]),
    ("openset", ["scripts.eval_openset"]),
    ("unknown", ["scripts.eval_unknown"]),
    ("robust", ["scripts.evaluate"]),
    ("optimize", ["scripts.optimize"]),
    ("report", ["scripts.write_report"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(a.start):]:
        if name in a.skip:
            continue
        print(f"\n===== {name}: python -m {' '.join(cmd)}", flush=True)
        subprocess.run([sys.executable, "-m", *cmd], check=True)
        if name == "clip":                     # deployable copy of the fine-tuned CLIP weights
            import shutil
            from src import config as C
            shutil.copy2(C.RUNS_DIR / "clip_ft" / "best_fp16.pt", C.MODELS_DIR / "clip_ft_fp16.pt")


if __name__ == "__main__":
    main()
