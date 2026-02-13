#!/usr/bin/env python3
"""
Unified generation runner for CreativityPrism.

Supported tasks:
- aut, ttcw, cshort      -> aut_ttcw_cshort
- ttct                   -> ttct
- creative_math          -> math_n_index
- creativity_index       -> math_n_index (book|poem|speech)
- neocoder               -> neocoder_dat (CodeForce/NeoCoder)
- dat                    -> neocoder_dat (Divergent Association Task)
- cs4                    -> cs4
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# For aut/ttcw/cshort (aut_ttcw_cshort)
AUT_TTCW_CSHORT_MODEL_MAP = {
    "gpt": "gpt.json",
    "claude": "claude.json",
    "gemini": "gemini.json",
    "qwen_72b": "qwen_72b.json",
    "qwen_32b": "qwen_32b.json",
    "llama_70b": "llama_70b.json",
    "mistral_small_24b": "mistral_small_24b.json",
    "mixtral_8x7b": "mixtral_8x7b.json",
    "deepseek_llama_70b": "deepseek_llama_70b.json",
    "deepseek_qwen_32b": "deepseek_qwen_32b.json",
}

AUT_TASK_DIR_KEY = {
    "aut": "aut",
    "ttcw": "ttcw",
    "cshort": "creative_short",
}

# Script paths
CS4_SCRIPT = "cs4/story_generator/cs4_story_generator.py"

# NeoCoder/DAT defaults
NEOCODER_DATASET_DEFAULT = "datasets/CodeForce/NeoCoder/NeoCoder.json"
NEOCODER_DP_ROUNDS_DEFAULT = 5
NEOCODER_BATCH_SIZE_DEFAULT = 1
DAT_REPEAT_DEFAULT = 100


def _default_out(task: str) -> Path:
    """Generate default output directory with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"{task}_{ts}"


def _ensure_outdir(path: Path):
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd, env=None, dry=False, cwd=None):
    """Execute command with optional dry-run mode."""
    print(f"[unified_generate] Working directory: {cwd or os.getcwd()}")
    print(f"[unified_generate] Command:\n  {' '.join(str(c) for c in cmd)}")
    if dry:
        print("[unified_generate] DRY RUN - command not executed")
        return 0
    return subprocess.call(cmd, env=env or os.environ.copy(), cwd=cwd)

def run_aut_ttcw_cshort(task: str, model: str, config_path: str, outdir: Path,
                        limit: int, seed: int, temperature: float,
                        max_new_tokens: int, dry: bool):
    """
    Run AUT/TTCW/Creative Short Story tasks.
    Delegates to aut_ttcw_cshort/run_inference.py.
    
    The script expects config files with structure:
    {
        "experiments_list": [{
            "task": "creative_writing" | "aut_push" | "creative_short",
            "run_id": "...",
            "model_name": "...",
            ...
        }]
    }
    
    Args:
        task: CLI task name (aut/ttcw/cshort) - maps to config directory
        model: Short model key (gpt, claude, etc.)
        config_path: Override with custom config path
        dry: Dry run mode
    """
    task_dir = AUT_TASK_DIR_KEY[task]
    
    if config_path:
        cfg = Path(config_path)
        if not cfg.is_absolute():
            cfg = REPO_ROOT / "aut_ttcw_cshort" / config_path
    else:
        name = AUT_TTCW_CSHORT_MODEL_MAP.get(model)
        if not name:
            raise ValueError(
                f"[aut_ttcw_cshort] Unknown model key: {model}. "
                f"Available: {list(AUT_TTCW_CSHORT_MODEL_MAP.keys())}"
            )
        cfg = REPO_ROOT / "aut_ttcw_cshort" / "configs" / task_dir / "inference" / name
    
    if not cfg.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg}\n"
            f"For task '{task}', expected at: aut_ttcw_cshort/configs/{task_dir}/inference/{model}.json"
        )
    
    # Validate config structure
    try:
        with open(cfg) as f:
            config_data = json.load(f)
        if 'experiments_list' not in config_data:
            raise ValueError(
                f"Config file {cfg} missing 'experiments_list' field. "
                f"Expected structure: {{'experiments_list': [{{'task': '...', 'run_id': '...', ...}}]}}"
            )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {cfg}: {e}")
    
    runner = REPO_ROOT / "aut_ttcw_cshort" / "run_inference.py"
    if not runner.exists():
        raise FileNotFoundError(f"Expected {runner} not found.")
    
    # Make config path relative to aut_ttcw_cshort/
    config_rel = cfg.relative_to(REPO_ROOT / "aut_ttcw_cshort")
    
    cmd = [sys.executable, str(runner.name), str(config_rel)]
    
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "aut_ttcw_cshort")


def run_math_n_index(task: str, subtask: str, config_path: str, 
                     experiment_index: int, run_all: bool, dry: bool):
    """
    Run creative_math or creativity_index tasks.
    Calls math_n_index/run_inference_all.py.
    
    The script expects:
    - config_file: JSON with 'experiments_list' array
    - experiment_index: Which experiment to run (0-based index)
    
    Config structure:
    {
        "experiments_list": [
            {
                "task": "creative_math" | "book" | "poem" | "speech",
                "model_name": "...",
                ...
            }
        ]
    }
    
    Args:
        task: "creative_math" or "creativity_index"
        subtask: For creativity_index: "book" | "poem" | "speech" (default: "book")
        config_path: Path to config JSON
        experiment_index: Which experiment to run (default: 0)
        run_all: If True, run all experiments in config sequentially
        dry: Dry run mode
    """
    script = REPO_ROOT / "math_n_index" / "run_inference_all.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected {script} not found.")
    
    # Resolve config file
    if config_path:
        cfg = Path(config_path)
        if not cfg.is_absolute():
            cfg = REPO_ROOT / "math_n_index" / config_path
    else:
        if task == "creative_math":
            cfg = REPO_ROOT / "math_n_index" / "configs" / "inference_creative_math.json"
        else:  # creativity_index
            subtask_name = subtask or "book"
            cfg = REPO_ROOT / "math_n_index" / "configs" / f"inference_creative_index_{subtask_name}.json"
    
    if not cfg.exists():
        raise FileNotFoundError(f"Config file not found: {cfg}")
    
    # Validate config structure
    try:
        with open(cfg) as f:
            config_data = json.load(f)
        if 'experiments_list' not in config_data:
            raise ValueError(
                f"Config file {cfg} missing 'experiments_list' field. "
                f"Expected structure: {{'experiments_list': [{{'task': '...', 'model_name': '...', ...}}]}}"
            )
        
        num_experiments = len(config_data['experiments_list'])
        if experiment_index >= num_experiments:
            raise ValueError(
                f"experiment_index={experiment_index} but config only has {num_experiments} experiments"
            )
        
        # Validate task field matches
        expected_task = subtask if task == "creativity_index" else task
        actual_task = config_data['experiments_list'][experiment_index].get('task')
        if actual_task != expected_task:
            print(
                f"WARNING: Config experiment[{experiment_index}] has task='{actual_task}' "
                f"but expected '{expected_task}' based on CLI args"
            )
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {cfg}: {e}")
    
    # Make config path relative to math_n_index/
    try:
        config_rel = cfg.relative_to(REPO_ROOT / "math_n_index")
    except ValueError:
        config_rel = cfg
    
    if run_all:
        # Run all experiments sequentially
        return_codes = []
        for i in range(num_experiments):
            print(f"\n[math_n_index] Running experiment {i+1}/{num_experiments}")
            cmd = [sys.executable, str(script.name), str(config_rel), str(i)]
            rc = _run(cmd, dry=dry, cwd=REPO_ROOT / "math_n_index")
            return_codes.append(rc)
            if rc != 0:
                print(f"WARNING: Experiment {i} failed with return code {rc}")
        return max(return_codes) if return_codes else 0
    else:
        cmd = [sys.executable, str(script.name), str(config_rel), str(experiment_index)]
        return _run(cmd, dry=dry, cwd=REPO_ROOT / "math_n_index")


def run_ttct(model: str, temperature: float, dry: bool):
    """
    Run TTCT task.
    Calls ttct/src/inference/__init__.py directly.
    
    The script generates responses for 3 prompt types (basic, instructive, cot)
    and saves to data/outputs/temp_{temp}/{model_name}.json
    
    Args:
        model: Model name (e.g., 'gpt-4-turbo', 'Qwen2.5-72B-Instruct')
        temperature: Sampling temperature (default: 1.0 in original paper)
        dry: Dry run mode
    """
    script = REPO_ROOT / "ttct" / "src" / "inference" / "__init__.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected {script} not found.")
    
    if temperature is None:
        temperature = 1.0
    
    cmd = [
        sys.executable, "src/inference/__init__.py",
        "-model_name", model,
        "-temp", str(int(temperature))
    ]

    return _run(cmd, dry=dry, cwd=REPO_ROOT / "ttct")


def run_neocoder(model_name: str, dataset_path: str, dp_rounds: int,
                batch_size: int, outdir: Path, temperature: float,
                top_p: float, max_tokens: int, overwrite: bool, dry: bool):
    """
    Run NeoCoder task (CodeForce competitive programming).
    Calls neocoder_dat/steps/inference_dp.py directly.
    
    Args:
        model_name: HuggingFace model name or API model name
        dataset_path: Path to NeoCoder dataset JSON
        dp_rounds: Number of divergent programming rounds
        batch_size: Batch size for inference
        outdir: Output directory for results
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        max_tokens: Maximum tokens to generate
        overwrite: Whether to overwrite existing results
        dry: Dry run mode
    """
    script = REPO_ROOT / "neocoder_dat" / "steps" / "inference_dp.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected {script} not found.")
    
    # Resolve dataset path
    if dataset_path:
        dataset = Path(dataset_path)
    else:
        dataset = REPO_ROOT / "neocoder_dat" / NEOCODER_DATASET_DEFAULT
    
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    
    _ensure_outdir(outdir)
    
    cmd = [
        sys.executable, "steps/inference_dp.py",
        "--dataset-path", str(dataset),
        "--model-name", model_name,
        "--dp-rounds", str(dp_rounds),
        "--batch-size", str(batch_size),
        "--output-dir", str(outdir),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-tokens", str(max_tokens),
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "neocoder_dat")


def run_dat(model_name: str, repeat: int, outdir: Path,
           temperature: float, top_p: float, max_tokens: int,
           overwrite: bool, dry: bool):
    """
    Run DAT (Divergent Association Task) - creative word generation.
    Calls neocoder_dat/steps/inference_dat.py directly.
    
    Args:
        model_name: HuggingFace model name or API model name
        repeat: Number of repetitions (samples to generate)
        outdir: Output directory for results
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        overwrite: Whether to overwrite existing results
        dry: Dry run mode
    """
    script = REPO_ROOT / "neocoder_dat" / "steps" / "inference_dat.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected {script} not found.")
    
    _ensure_outdir(outdir)
    
    cmd = [
        sys.executable, "steps/inference_dat.py",
        "--model-name", model_name,
        "--repeat", str(repeat),
        "--output-dir", str(outdir),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--max-tokens", str(max_tokens),
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "neocoder_dat")


def run_cs4(config_path: str, dry: bool):
    """
    Run CS4 task.
    Calls cs4/story_generator/cs4_story_generator.py.
    
    CS4 uses config files to control all parameters including:
    - Model selection (model_name in experiments_list)
    - Temperature, top_p (in config)
    - Output directory (run_id in config -> outputs to ./output/{run_id}/)
    
    Args:
        config_path: Path to JSON config file
        dry: Dry run mode
    """
    py = REPO_ROOT / CS4_SCRIPT
    if not py.exists():
        raise FileNotFoundError(f"Expected {py} not found.")
    
    if not config_path:
        raise ValueError("CS4 requires --config parameter. "
                        "Example: --config configs/llama_70b.json")
    
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / "cs4" / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Make config path relative to cs4/ if possible
    try:
        config_rel = config_file.relative_to(REPO_ROOT / "cs4")
    except ValueError:
        config_rel = config_file  # Use absolute if outside cs4/
    
    cmd = [sys.executable, "story_generator/cs4_story_generator.py", str(config_rel)]
    
    return _run(cmd, dry=dry, cwd=REPO_ROOT / "cs4")

def main():
    p = argparse.ArgumentParser(
        description="Unified generator for CreativityPrism tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NeoCoder with open-source model
  python unified_generate.py --task neocoder \\
      --model allenai/OLMo-2-1124-13B-SFT \\
      --dp-rounds 5 --temperature 0.75

  # NeoCoder with API model
  python unified_generate.py --task neocoder \\
      --model gpt-4-turbo \\
      --dp-rounds 3 --temperature 0.8

  # DAT task
  python unified_generate.py --task dat \\
      --model gpt-4-turbo \\
      --repeat 100 --temperature 0.75

  # AUT task
  python unified_generate.py --task aut --model gpt
  
  # Creative Math - run all experiments
  python unified_generate.py --task creative_math \\
      --config configs/inference_creative_math.json \\
      --run-all-experiments
"""
    )
    
    # Core arguments
    p.add_argument("--task", required=True,
                   choices=["aut", "ttcw", "cshort", "ttct", 
                           "creative_math", "creativity_index",
                           "neocoder", "dat", "cs4"],
                   help="Task to run")
    
    p.add_argument("--model", default=None,
                   help="Model name. For neocoder/dat/ttct: full HF name or API model. "
                        "For aut/ttcw/cshort: short key (gpt, claude, etc.)")
    
    p.add_argument("--config", default=None,
                   help="Path to config JSON. "
                        "For cs4: required. "
                        "For aut/ttcw/cshort/math_n_index: optional, overrides defaults.")
    
    p.add_argument("--out", dest="outdir", default=None,
                   help="Output directory (default: runs/<task>_<timestamp>)")
    
    # Generic parameters
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples (if supported by task)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (if supported by task)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (default varies by task)")
    p.add_argument("--top-p", type=float, default=None,
                   help="Top-p (nucleus) sampling (default varies by task)")
    p.add_argument("--max-tokens", "--max-new-tokens", type=int, default=None,
                   dest="max_tokens",
                   help="Maximum tokens to generate (default varies by task)")
    
    # Task-specific arguments
    p.add_argument("--subtask", default=None,
                   help="[creativity_index] Subtask: book|poem|speech")
    p.add_argument("--experiment-index", type=int, default=0,
                   help="[math_n_index] Which experiment to run from config (default: 0)")
    p.add_argument("--run-all-experiments", action="store_true",
                   help="[math_n_index] Run all experiments in config sequentially")
    
    # NeoCoder specific
    p.add_argument("--dataset-path", default=None,
                   help="[neocoder] Path to dataset (default: datasets/CodeForce/NeoCoder/NeoCoder.json)")
    p.add_argument("--dp-rounds", type=int, default=None,
                   help=f"[neocoder] Number of DP rounds (default: {NEOCODER_DP_ROUNDS_DEFAULT})")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"[neocoder] Batch size (default: {NEOCODER_BATCH_SIZE_DEFAULT})")
    
    # DAT specific
    p.add_argument("--repeat", type=int, default=None,
                   help=f"[dat] Number of repetitions (default: {DAT_REPEAT_DEFAULT})")
    
    # Utility
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--dry-run", action="store_true",
                   help="Print command without executing")
    
    args = p.parse_args()
    
    # Set defaults
    outdir = Path(args.outdir) if args.outdir else _default_out(args.task)
    
    # Add repo root to Python path
    sys.path.insert(0, str(REPO_ROOT))
    
    # Route to appropriate task runner
    try:
        if args.task in ("aut", "ttcw", "cshort"):
            if not args.model:
                args.model = "gpt"
            return sys.exit(run_aut_ttcw_cshort(
                task=args.task,
                model=args.model,
                config_path=args.config,
                outdir=outdir,
                limit=args.limit,
                seed=args.seed,
                temperature=args.temperature or 0.7,
                max_new_tokens=args.max_tokens or 512,
                dry=args.dry_run,
            ))
        
        elif args.task in ("creative_math", "creativity_index"):
            return sys.exit(run_math_n_index(
                task=args.task,
                subtask=args.subtask,
                config_path=args.config,
                experiment_index=args.experiment_index,
                run_all=args.run_all_experiments,
                dry=args.dry_run,
            ))
        
        elif args.task == "ttct":
            if not args.model:
                raise ValueError("TTCT requires --model parameter. "
                                "Examples: gpt-4-turbo, Qwen2.5-72B-Instruct")
            
            return sys.exit(run_ttct(
                model=args.model,
                temperature=args.temperature,
                dry=args.dry_run,
            ))
        
        elif args.task == "neocoder":
            if not args.model:
                raise ValueError("--model is required for neocoder task. "
                               "Examples: gpt-4-turbo, allenai/OLMo-2-1124-13B-SFT")
            return sys.exit(run_neocoder(
                model_name=args.model,
                dataset_path=args.dataset_path,
                dp_rounds=args.dp_rounds or NEOCODER_DP_ROUNDS_DEFAULT,
                batch_size=args.batch_size or NEOCODER_BATCH_SIZE_DEFAULT,
                outdir=outdir,
                temperature=args.temperature or 0.75,
                top_p=args.top_p or 1.0,
                max_tokens=args.max_tokens or 1024,
                overwrite=args.overwrite,
                dry=args.dry_run,
            ))
        
        elif args.task == "dat":
            if not args.model:
                raise ValueError("--model is required for dat task. "
                               "Examples: gpt-4-turbo, meta-llama/Llama-3.1-8B-Instruct")
            return sys.exit(run_dat(
                model_name=args.model,
                repeat=args.repeat or DAT_REPEAT_DEFAULT,
                outdir=outdir,
                temperature=args.temperature or 0.75,
                top_p=args.top_p or 1.0,
                max_tokens=args.max_tokens or 4096,
                overwrite=args.overwrite,
                dry=args.dry_run,
            ))
        
        elif args.task == "cs4":
            if not args.config:
                raise ValueError("CS4 requires --config parameter. "
                                "Example: python unified_generate.py --task cs4 "
                                "--config configs/llama_70b.json")
            return sys.exit(run_cs4(
                config_path=args.config,
                dry=args.dry_run,
            ))
        
        else:
            raise ValueError(f"Unhandled task: {args.task}")
    
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()