#!/usr/bin/env python3
"""
Unified evaluation runner for CreativityPrism.

Supported tasks:
- aut, ttcw, cshort      -> aut_ttcw_cshort
- ttct                   -> ttct
- creative_math          -> math_n_index
- creativity_index       -> math_n_index (book|poem|speech)
- neocoder               -> neocoder_dat (CodeForce/NeoCoder)
- dat                    -> neocoder_dat (Divergent Association Task)
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def _run(cmd, env=None, dry=False, cwd=None):
    """Execute command with optional dry-run mode."""
    print(f"[unified_evaluate] Working directory: {cwd or os.getcwd()}")
    print(f"[unified_evaluate] Command:\n  {' '.join(str(c) for c in cmd)}")
    if dry:
        print("[unified_evaluate] DRY RUN - command not executed")
        return 0
    return subprocess.call(cmd, env=env or os.environ.copy(), cwd=cwd)

def run_aut_ttcw_cshort(config_path: str, dry: bool):
    """
    Run evaluation for AUT/TTCW/Creative Short Story tasks.
    Delegates to aut_ttcw_cshort/run_evaluation.py.
    """
    runner = REPO_ROOT / "tasks" / "aut_ttcw_cshort" / "run_evaluation.py"
    if not runner.exists():
        raise FileNotFoundError(f"Expected {runner} not found.")
    
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = REPO_ROOT / "tasks" / "aut_ttcw_cshort" / config_path
    
    if not cfg.exists():
        raise FileNotFoundError(f"Config file not found: {cfg}")

    # Make config path relative to aut_ttcw_cshort/
    try:
        config_rel = cfg.relative_to(REPO_ROOT / "tasks" / "aut_ttcw_cshort")
    except ValueError:
        config_rel = cfg

    cmd = [sys.executable, str(runner.name), str(config_rel)]
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "tasks" / "aut_ttcw_cshort")

def run_math_n_index(task: str, model_to_evaluate: str, portion: float, dry: bool):
    """
    Run evaluation for creative_math or creativity_index tasks.
    Delegates to math_n_index/src/evaluation/evaluation.py or similar.
    """
    # Note: math_n_index has multiple evaluation scripts. 
    # Defaulting to the main evaluation.py for now.
    script = REPO_ROOT / "tasks" / "math_n_index" / "src" / "evaluation" / "evaluation.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected {script} not found.")
    
    cmd = [
        sys.executable, "src/evaluation/evaluation.py",
        "--model_to_evaluate", model_to_evaluate,
        "--portion", str(portion)
    ]
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "tasks" / "math_n_index")

def run_ttct(model_name: str, eval_model: str, temp: int, dry: bool):
    """
    Run evaluation for TTCT task.
    Delegates to ttct/src/evaluation/ttct_evaluation.py.
    """
    script = REPO_ROOT / "tasks" / "ttct" / "src" / "evaluation" / "ttct_evaluation.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected {script} not found.")
    
    # Based on ttct_evaluation.py arguments
    cmd = [
        sys.executable, "src/evaluation/ttct_evaluation.py",
        "-model_name", model_name,
        "-eval_model", eval_model,
        "-temp", str(temp)
    ]
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "tasks" / "ttct")

def run_neocoder(task: str, result_path: str, test_case_path: str, human_path: str, dry: bool):
    """
    Run evaluation for NeoCoder or DAT tasks.
    """
    if task == "dat":
        script = REPO_ROOT / "tasks" / "neocoder_dat" / "steps" / "evaluate_dat.py"
        cmd = [sys.executable, "steps/evaluate_dat.py", "--result-path", result_path]
    else:
        script = REPO_ROOT / "tasks" / "neocoder_dat" / "steps" / "evaluate_neogauge.py"
        cmd = [
            sys.executable, "steps/evaluate_neogauge.py",
            "--task", task,
            "--inference-result-path", result_path
        ]
        if test_case_path:
            cmd.extend(["--test-case-path", test_case_path])
        if human_path:
            cmd.extend(["--human-solution-path", human_path])

    return _run(cmd, dry=dry, cwd=REPO_ROOT / "tasks" / "neocoder_dat")

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner for CreativityPrism")
    subparsers = parser.add_subparsers(dest="task", help="Task to evaluate")

    # AUT/TTCW/CSHORT
    aut_parser = subparsers.add_parser("aut", help="AUT/TTCW/Creative Short Story evaluation")
    aut_parser.add_argument("--config", type=str, required=True, help="Path to evaluation config JSON")

    # TTCT
    ttct_parser = subparsers.add_parser("ttct", help="TTCT evaluation")
    ttct_parser.add_argument("--model_name", type=str, required=True, help="Model to evaluate")
    ttct_parser.add_argument("--eval_model", type=str, default="gpt-4", help="Model to use as judge")
    ttct_parser.add_argument("--temp", type=int, default=1, help="Temperature used during inference")

    # Math & Index
    math_parser = subparsers.add_parser("math", help="Creative Math / Creativity Index evaluation")
    math_parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    math_parser.add_argument("--portion", type=float, default=1.0, help="Portion of data to evaluate")

    # NeoCoder & DAT
    neo_parser = subparsers.add_parser("neocoder", help="NeoCoder / DAT evaluation")
    neo_parser.add_argument("--subtask", choices=["correctness", "detection", "creativity", "dat"], required=True)
    neo_parser.add_argument("--result_path", type=str, required=True, help="Path to inference results")
    neo_parser.add_argument("--test_case_path", type=str, help="Path to test cases (for correctness)")
    neo_parser.add_argument("--human_path", type=str, help="Path to human solutions (for detection/creativity)")

    parser.add_argument("--dry-run", action="store_true", help="Don't actually run commands")

    args = parser.parse_args()

    if not args.task:
        parser.print_help()
        return

    try:
        if args.task == "aut":
            run_aut_ttcw_cshort(args.config, args.dry_run)
        elif args.task == "ttct":
            run_ttct(args.model_name, args.eval_model, args.temp, args.dry_run)
        elif args.task == "math":
            run_math_n_index(args.task, args.model, args.portion, args.dry_run)
        elif args.task == "neocoder":
            run_neocoder(args.subtask, args.result_path, args.test_case_path, args.human_path, args.dry_run)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
