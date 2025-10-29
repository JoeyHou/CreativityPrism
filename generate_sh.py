#!/usr/bin/env python3
"""
Unified generation runner for CreativityPrism.

Goals
- One CLI to generate responses for any task in this repo, while
  reusing each subproject's *existing* prompt builders & dataset loaders.
- Zero prompt reimplementation: we always delegate to the project's own
  run_* scripts or inference drivers.

Supported task keys
- aut            -> aut_ttcw_cshort (AUT / alternative uses)
- ttcw           -> aut_ttcw_cshort (TTCW creative writing)
- cshort         -> aut_ttcw_cshort (Creative Short Story)
- ttct           -> ttct
- creative_math  -> math_n_index
- creativity_index -> math_n_index (subtypes: book|poem|speech)
- neocoder       -> neocoder_dat (codeforces / NeoCoder)
- dat            -> neocoder_dat (Divergent Association Task coding variant)

Usage examples
- python unified_generate.py --task aut --model gpt --out runs/aut_gpt
- python unified_generate.py --task ttcw --model qwen_72b
- python unified_generate.py --task cshort --model llama_70b
- python unified_generate.py --task creative_math --model gpt --limit 200
- python unified_generate.py --task creativity_index --subtask book --model gpt
- python unified_generate.py --task ttct --model gpt
- python unified_generate.py --task neocoder --model llama_70b
- python unified_generate.py --task dat --model gpt

Notes
- API keys are picked up by the subprojects (e.g., OPENAI_API_KEY,
  ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.).
- If you pass --config with a path to a JSON, that overrides --model.
- --dry-run prints the exact command it would execute.

"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# -------- Model config lookup (defaults pulled from your repo layout) --------
# For aut/ttcw/cshort (aut_ttcw_cshort)
AUT_TTCW_CSHORT_MODEL_MAP = {
    # These map to files under aut_ttcw_cshort/configs/<task>/inference/
    # You can add/remove entries as needed.
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

# For math_n_index (creative_math & creativity_index)
MATH_INDEX_DEFAULT_CONFIG = {
    "creative_math": "math_n_index/configs/inference_creative_math.json",
    # The creativity_index has separate files for each subtask; default to book.
    "creativity_index:book": "math_n_index/configs/inference_creative_index_book.json",
    "creativity_index:poem": "math_n_index/configs/inference_creative_index_book.json",   # adjust if you add poem/speech configs later
    "creativity_index:speech": "math_n_index/configs/inference_creative_index_book.json",
}

# ttct has its own mini project; we call its script which uses its own loaders.
TTCT_SCRIPT = "ttct/scripts/run_inference.sh"

# neocoder/dat have their own shell wrappers too.
NEOCODER_SCRIPT = "neocoder_dat/scripts/inference_neocoder.sh"
DAT_SCRIPT      = "neocoder_dat/scripts/inference_dat.sh"


def _default_out(task: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"{task}_{ts}"


def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd, env=None, dry=False):
    print("[unified_generate] command:\n  " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd, env=env or os.environ.copy())


# ----------------------------- Runners ---------------------------------

def run_aut_ttcw_cshort(task: str, model: str, config_path: str, outdir: Path,
                        limit: int, seed: int, temperature: float,
                        max_new_tokens: int, dry: bool):
    """
    Delegates to aut_ttcw_cshort/run_inference.py (keeps that project's
    prompt formatting intact).
    """
    task_dir = AUT_TASK_DIR_KEY[task]  # aut | ttcw | creative_short
    if config_path:
        cfg = config_path
    else:
        # Resolve model -> config file under the correct task's inference dir
        name = AUT_TTCW_CSHORT_MODEL_MAP.get(model)
        if not name:
            raise ValueError(f"[aut_ttcw_cshort] Unknown model key: {model}")
        cfg = REPO_ROOT / "aut_ttcw_cshort" / "configs" / task_dir / "inference" / name

    runner = REPO_ROOT / "aut_ttcw_cshort" / "run_inference.py"
    if not runner.exists():
        raise FileNotFoundError("Expected aut_ttcw_cshort/run_inference.py not found.")

    cmd = [sys.executable, str(runner),
           "--config", str(cfg),
           "--task", task]  # the underlying script distinguishes templates by task

    # Optional knobs: only add if provided; the subproject ignores unknown flags.
    if outdir:
        cmd += ["--output_dir", str(outdir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(max_new_tokens)]

    return _run(cmd, dry=dry)


def run_math_n_index(task: str, subtask: str, config_path: str, outdir: Path,
                     limit: int, seed: int, temperature: float,
                     max_new_tokens: int, dry: bool):
    """
    Delegates to math_n_index/run_inference.py.
    creative_math uses its own prompt_engineering module.
    creativity_index supports subtask: book|poem|speech.
    """
    runner = REPO_ROOT / "math_n_index" / "run_inference.py"
    if not runner.exists():
        raise FileNotFoundError("Expected math_n_index/run_inference.py not found.")

    if config_path:
        cfg = config_path
    else:
        key = task if task == "creative_math" else f"{task}:{subtask or 'book'}"
        default_cfg = MATH_INDEX_DEFAULT_CONFIG.get(key)
        if not default_cfg:
            raise ValueError(f"[math_n_index] No default config for {key}. Use --config.")
        cfg = REPO_ROOT / default_cfg

    cmd = [sys.executable, str(runner),
           "--task", task]
    if subtask:
        cmd += ["--subtask", subtask]
    cmd += ["--config", str(cfg)]

    if outdir:
        cmd += ["--output_dir", str(outdir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(max_new_tokens)]

    return _run(cmd, dry=dry)


def run_ttct(model: str, config_path: str, outdir: Path,
             limit: int, seed: int, temperature: float,
             max_new_tokens: int, dry: bool):
    """
    TTCT has its own mini-repo. Use its provided shell script so it can
    load prompts/datasets internally exactly as designed.
    """
    sh = REPO_ROOT / TTCT_SCRIPT
    if not sh.exists():
        # Fallback: try calling its Python driver if you prefer later.
        raise FileNotFoundError("Expected ttct/scripts/run_inference.sh not found.")

    # Many TTCT scripts are model-agnostic & pull model from a config or env.
    cmd = ["bash", str(sh)]
    if config_path:
        cmd += ["--config", str(config_path)]
    if outdir:
        cmd += ["--output_dir", str(outdir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(max_new_tokens)]
    if model:
        cmd += ["--model", model]

    return _run(cmd, dry=dry)


def run_neocoder_or_dat(which: str, outdir: Path, limit: int, seed: int,
                        temperature: float, max_new_tokens: int, dry: bool):
    """
    For NeoCoder/DAT, call their provided shell wrappers so all in-repo
    prompt formatting is reused.
    """
    sh = REPO_ROOT / (NEOCODER_SCRIPT if which == "neocoder" else DAT_SCRIPT)
    if not sh.exists():
        raise FileNotFoundError(f"Expected script not found: {sh}")

    cmd = ["bash", str(sh)]
    if outdir:
        cmd += ["--output_dir", str(outdir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(max_new_tokens)]

    return _run(cmd, dry=dry)


# ----------------------------- CLI -------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Unified generator for CreativityPrism tasks (delegates to each subproject's own generators)."
    )
    p.add_argument("--task", required=True,
                   choices=["aut", "ttcw", "cshort",
                            "ttct", "creative_math", "creativity_index",
                            "neocoder", "dat"],
                   help="Which task to run.")
    p.add_argument("--subtask", default=None,
                   help="Optional subtask (e.g., for creativity_index: book|poem|speech).")
    p.add_argument("--model", default="gpt",
                   help="Short model key (used to pick a config in aut/ttcw/cshort). Ignored if --config is given.")
    p.add_argument("--config", default=None,
                   help="Path to a model config JSON (overrides --model).")
    p.add_argument("--out", dest="outdir", default=None,
                   help="Output directory. Defaults to runs/<task>_<timestamp>.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional: cap number of items for a quick run.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print the command without executing.")
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else _default_out(args.task)
    _ensure_outdir(outdir)

    # make imports available if any runner imports modules
    sys.path.insert(0, str(REPO_ROOT))

    if args.task in ("aut", "ttcw", "cshort"):
        return sys.exit(run_aut_ttcw_cshort(
            task=args.task,
            model=args.model,
            config_path=args.config,
            outdir=outdir,
            limit=args.limit,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dry=args.dry_run,
        ))

    if args.task in ("creative_math", "creativity_index"):
        return sys.exit(run_math_n_index(
            task=args.task,
            subtask=args.subtask,
            config_path=args.config,
            outdir=outdir,
            limit=args.limit,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dry=args.dry_run,
        ))

    if args.task == "ttct":
        return sys.exit(run_ttct(
            model=args.model,
            config_path=args.config,
            outdir=outdir,
            limit=args.limit,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dry=args.dry_run,
        ))

    if args.task in ("neocoder", "dat"):
        return sys.exit(run_neocoder_or_dat(
            which=args.task,
            outdir=outdir,
            limit=args.limit,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dry=args.dry_run,
        ))

    raise ValueError(f"Unhandled task: {args.task}")


if __name__ == "__main__":
    main()
