"""
run_all.py
==========
Run all simulation scripts for P12 in sequence.

Article : "Massive AI Model Orchestration for 6G" (IEEE Wireless Communications)
Purpose : Execute scripts 01-08 and report pass/fail status for each.

Usage:
    python run_all.py
"""

import subprocess
import sys
import os

SCRIPTS = [
    "script_01_main_comparison.py",
    "script_02_handover_prediction_lstm.py",
    "script_03_channel_estimation_vit.py",
    "script_04_pareto_optimization.py",
    "script_05_carbon_aware_scheduling.py",
    "script_06_early_exit_networks.py",
    "script_07_split_computing.py",
    "script_08_kpi_summary.py",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print("="*60)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


def main():
    print("P12 Simulation Suite – Massive AI Model Orchestration for 6G")
    print("=" * 60)

    results = {}
    for script in SCRIPTS:
        results[script] = run_script(script)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = True
    for script, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] {script}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("All scripts completed successfully.")
    else:
        print("Some scripts failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
