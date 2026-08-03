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

# Derived from the script's own location, so a script generated on a laptop still
# works on the cluster: no absolute path is baked in.
cd "$(dirname "$(readlink -f "$0")")/../.."

# The job re-invokes the runner without --slurm, so artifact materialization and
# metadata.json are produced inside the job exactly as they are for a local run.
{{COMMAND}}
