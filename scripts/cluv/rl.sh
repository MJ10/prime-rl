#!/bin/bash
# Single-node `rl` job for `cluv submit`. The config must not have a `[slurm]` block:
# prime-rl runs trainer, inference and orchestrator inside this allocation.
#
#   cluv submit mila scripts/cluv/rl.sh -- examples/glyphbench/single_node.toml [rl overrides...]
#
# Runs from a per-commit snapshot of the cluster checkout so later `cluv sync`s don't change the
# code under a queued, running or requeued job. Requeued jobs resume from their latest checkpoint.
#
# On clusters whose compute nodes have no internet (UV_OFFLINE=1), build the snapshot and download
# the model on a login node first, from the cluster checkout:
#
#   GIT_COMMIT=$(git rev-parse HEAD) bash scripts/cluv/rl.sh --prepare <config.toml>
#
# CPUs and memory are set per cluster in [tool.cluv.clusters] since node shapes differ.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
set -euo pipefail

prepare=false
if [ "${1:-}" = "--prepare" ]; then
    prepare=true
    shift
fi
: "${GIT_COMMIT:?GIT_COMMIT is not set. Submit this script with 'cluv submit'.}"
config=${1:?usage: rl.sh [--prepare] <config.toml> [rl overrides...]}
shift

project_root=$PWD
prime_rl_root=${PRIME_RL_ROOT:-$SCRATCH/prime-rl}
snapshot=$prime_rl_root/snapshots/$GIT_COMMIT
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/.cache/uv}
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export PRL_OUTPUT_DIR=${PRL_OUTPUT_DIR:-$prime_rl_root/outputs}

# Build the snapshot once per commit. Submodules are cloned from the cluster checkout, whose
# submodules `cluv sync` keeps current (see scripts/cluv/README.md).
mkdir -p "$prime_rl_root/snapshots"
(
    flock 9
    if [ ! -e "$snapshot/.ready" ]; then
        if [ "${UV_OFFLINE:-0}" = 1 ]; then
            echo "No snapshot for $GIT_COMMIT and no internet. Run 'rl.sh --prepare' on a login node." >&2
            exit 1
        fi
        rm -rf "$snapshot"
        # Compute nodes may lack git-lfs; leave LFS pointer files (none of which rl needs) as they are.
        export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0= \
            GIT_CONFIG_KEY_1=filter.lfs.required GIT_CONFIG_VALUE_1=false
        git clone --quiet --no-checkout "$project_root" "$snapshot"
        git -C "$snapshot" checkout --quiet --detach "$GIT_COMMIT"
        git -C "$snapshot" config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
            while read -r key path; do
                [ -e "$project_root/$path/.git" ] || continue
                name=${key#submodule.}
                name=${name%.path}
                git -C "$snapshot" submodule --quiet init -- "$path"
                git -C "$snapshot" config "submodule.$name.url" "$project_root/$path"
            done
        git -C "$snapshot" -c protocol.file.allow=always submodule update --quiet
        # Mooncake (KV-cache offload, unused here) only has glibc 2.35+ wheels; DRAC nodes have 2.34.
        uv --directory "$snapshot" sync --all-extras --locked --no-install-package mooncake-transfer-engine
        touch "$snapshot/.ready"
    fi
) 9>"$snapshot.lock"

cd "$snapshot"

if $prepare; then
    model=$(uv run --no-sync python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["model"]["name"])' "$config")
    uv run --no-sync hf download "$model"
    echo "Prepared $snapshot with $model"
    exit 0
fi

# The job id survives requeues, so a requeued job finds its own run directory.
run_name=$(basename "$config" .toml)-$SLURM_JOB_ID
run_dir=$PRL_OUTPUT_DIR/$run_name
if compgen -G "$run_dir/checkpoints/step_*" >/dev/null; then
    run_mode=(--resume)
else
    run_mode=(--clean)
fi

# Nodes can be shared with other jobs, so give each job its own ports instead of the fixed
# defaults. Assumes an inference server and NCCL weight broadcast, as single-node configs use.
port=$((12000 + SLURM_JOB_ID % 800 * 10))
ports=(
    --inference.server.port "$port"
    --inference.backend-port $((port + 1))
    --orchestrator.model.client.base-url "http://localhost:$port/v1"
    --rollout-transport.port $((port + 2))
    --weight-broadcast.port $((port + 3))
    --inference.vllm.data-parallel-rpc-port $((port + 4))
)

echo "commit=$GIT_COMMIT snapshot=$snapshot run_dir=$run_dir restart=${SLURM_RESTART_COUNT:-0} port=$port"
exec uv run --no-sync rl @ "$config" --run.name "$run_name" --no-dashboard "${run_mode[@]}" "${ports[@]}" "$@"
