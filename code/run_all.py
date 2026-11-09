"""run_all.py -- reproduce every figure and every number quoted in the report.

    python code/run_all.py   (run from the repository root)

Runtime: a few minutes (the cost-accuracy sweep includes sparse LU at
N=1024; the solver race runs 2e5-iteration Jacobi sweeps at N=64).
Requirements: numpy, scipy, matplotlib  (see requirements.txt).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = [
    "exp01_row_verification.py",
    "exp02_convergence_tables.py",
    "exp03_error_maps.py",
    "exp04_matrix_structure.py",
    "exp05_solvers_cost.py",
    "exp06_pure_neumann.py",
    "exp07_falsification.py",
    "exp08_tolerance_sweep.py",
]


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "code") + os.pathsep + env.get("PYTHONPATH", "")
    failures = []
    for script in EXPERIMENTS:
        path = os.path.join(ROOT, "experiments", script)
        print(f"\n########## {script} ##########")
        res = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
        if res.returncode != 0:
            failures.append(script)
    print("\n########## tests/test_smoke.py ##########")
    res = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "test_smoke.py")],
                         cwd=ROOT, env=env)
    if res.returncode != 0:
        failures.append("tests/test_smoke.py")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("\nAll experiments completed. Figures in figures/, numbers in figures/results.json")


if __name__ == "__main__":
    main()
