#!/bin/bash
{{SBATCH_DIRECTIVES}}
#~ Lines starting with #~ document the template and are stripped on render.
#~ The #SBATCH lines below are DEFAULTS ONLY; the resolved header replaces them:
#~   template defaults <- registry/tasks/{name}.yaml `slurm:` <- --slurm-override key=value
#~ A directive that resolves to an empty value is dropped, so `--slurm-override gres=`
#~ turns a GPU job into a CPU-only job -- worth doing for API-model runs, which
#~ never touch the GPU. --output/--error default to
#~ slurm_scripts/{label}/logs/%x-%j.{out,err}.
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

# Locating the repo from "$0" alone does NOT work under sbatch: SLURM copies the
# batch script into a node-local spool directory, so inside the job "$0" is
# /var/spool/slurmd/.../slurm_script and the derived root is /var/spool. Verified
# on the Pitt CRC cluster, 2026-08-04. Candidates are therefore tried in order and
# each must actually contain task_runner/run.py; {{REPO_ROOT}} is the generation-time
# absolute path, used only as a last resort so a relocated checkout still wins.
_cp_repo_root=""
for _cp_cand in "${CREATIVITYPRISM_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
                "$(dirname "$(readlink -f "$0")")/../.." "{{REPO_ROOT}}"; do
    if [[ -n "$_cp_cand" && -f "$_cp_cand/task_runner/run.py" ]]; then
        _cp_repo_root="$_cp_cand"; break
    fi
done
if [[ -z "$_cp_repo_root" ]]; then
    echo "could not locate the CreativityPrism repo root from inside the job." >&2
    echo "Set CREATIVITYPRISM_REPO_ROOT to the checkout and resubmit." >&2
    exit 3
fi
cd "$_cp_repo_root"

# Site-local setup (module load, PATH, LD_LIBRARY_PATH). The runner itself needs
# pyyaml, and it starts before any adapter -- so a cluster whose default python3
# lacks it must fix PATH here, not in the adapter's copy of this hook.
# Template: registry/environments/cluster_env.sh.example
if [[ -f registry/environments/.cluster_env.sh ]]; then
    source registry/environments/.cluster_env.sh
fi

# The job re-invokes the runner without --slurm, so artifact materialization and
# metadata.json are produced inside the job exactly as they are for a local run.
{{COMMAND}}
