#!/bin/bash
# Single-node `rl` job for `cluv submit`. The config must not have a `[slurm]` block:
# prime-rl runs trainer, inference and orchestrator inside this allocation.
#
#   cluv submit mila scripts/cluv/rl.sh -- examples/glyphbench/single_node.toml [rl overrides...]
#
# Runs from a per-commit snapshot of the cluster checkout so later `cluv sync`s don't change the
# code under a queued, running or requeued job, with its environment installed on node-local disk.
# Requeued jobs resume from their latest checkpoint. Logs are written on node-local disk and
# mirrored into the run directory every minute.
#
# On clusters whose compute nodes have no internet (UV_OFFLINE=1), fill the uv cache and download
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
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export PRL_OUTPUT_DIR=${PRL_OUTPUT_DIR:-$prime_rl_root/outputs}
# Each job installs its environment on node-local disk: importing vLLM from network storage is
# slow and has failed vLLM startup on Mila. PRIME_RL_LOCAL_UV_CACHE=1 also keeps the uv cache there,
# for filesystems whose file locks don't work across nodes (Mila's BeeGFS), where concurrent jobs
# corrupted a shared cache. Offline clusters read the shared cache that --prepare filled.
if $prepare; then
    export UV_PROJECT_ENVIRONMENT=$(mktemp -d "$prime_rl_root/venv.XXXXXX")
else
    export UV_PROJECT_ENVIRONMENT=$SLURM_TMPDIR/venv
fi
if ! $prepare && [ "${PRIME_RL_LOCAL_UV_CACHE:-0}" = 1 ]; then
    export UV_CACHE_DIR=$SLURM_TMPDIR/uv-cache
else
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/.cache/uv}
fi
# Mooncake (KV-cache offload, unused here) only has glibc 2.35+ wheels; DRAC nodes have 2.34.
sync_args=(--all-extras --locked --no-install-package mooncake-transfer-engine)

# Snapshot the commit once. Each job clones into a private directory and renames it into place,
# since file locks don't work across nodes on every cluster filesystem. Submodules are cloned from
# the cluster checkout, whose submodules `cluv sync` keeps current (see scripts/cluv/README.md).
if [ ! -d "$snapshot" ]; then
    mkdir -p "$prime_rl_root/snapshots"
    tmp=$(mktemp -d "$snapshot.tmp.XXXXXX")
    # Compute nodes may lack git-lfs; leave LFS pointer files (none of which rl needs) as they are.
    export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0= \
        GIT_CONFIG_KEY_1=filter.lfs.required GIT_CONFIG_VALUE_1=false
    git clone --quiet --no-checkout "$project_root" "$tmp"
    git -C "$tmp" checkout --quiet --detach "$GIT_COMMIT"
    git -C "$tmp" config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
        while read -r key path; do
            [ -e "$project_root/$path/.git" ] || continue
            name=${key#submodule.}
            name=${name%.path}
            git -C "$tmp" submodule --quiet init -- "$path"
            git -C "$tmp" config "submodule.$name.url" "$project_root/$path"
        done
    git -C "$tmp" -c protocol.file.allow=always submodule update --quiet
    mv -T "$tmp" "$snapshot" 2>/dev/null || rm -rf "$tmp"
fi

cd "$snapshot"
uv sync "${sync_args[@]}" --quiet

if $prepare; then
    model=$(uv run --no-sync python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["model"]["name"])' "$config")
    uv run --no-sync hf download "$model"
    rm -rf "$UV_PROJECT_ENVIRONMENT"
    echo "Prepared $GIT_COMMIT with $model"
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

# Logs go to node-local disk and are mirrored into $run_dir/logs every minute and on exit: on
# Mila's BeeGFS, appends to files held open for hours start failing with "Remote I/O error".
last_attempt=0
if [ "${run_mode[0]}" = --resume ]; then
    for dir in "$run_dir"/logs/attempt_* "$run_dir"/configs/attempt_*; do
        [ -e "$dir" ] || continue
        if [ "${dir##*_}" -gt "$last_attempt" ]; then
            last_attempt=${dir##*_}
        fi
    done
fi
attempt=attempt_$((last_attempt + 1))
local_logs=$SLURM_TMPDIR/logs/$attempt
export PRL_ATTEMPT_LOG_DIR=$local_logs PRL_ATTEMPT_CONFIG_DIR=$run_dir/configs/$attempt/resolved
mirror_logs() {
    mkdir -p "$run_dir/logs" &&
        rsync -a "$local_logs" "$run_dir/logs/" &&
        ln -sfn "$attempt" "$run_dir/logs/latest" &&
        ln -sfn "$attempt" "$run_dir/configs/latest"
}

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

echo "commit=$GIT_COMMIT snapshot=$snapshot run_dir=$run_dir logs=$run_dir/logs/$attempt restart=${SLURM_RESTART_COUNT:-0} port=$port"
# The first mirror waits a minute, so `--clean` has already reset the run directory.
(while sleep 60; do mirror_logs || true; done) &
mirror_pid=$!
trap 'kill "$mirror_pid" 2>/dev/null || true; mirror_logs' EXIT
# Slurm sends SIGTERM on preemption and cancellation: exit, mirroring the logs, once rl stops.
trap 'exit 143' TERM
uv run --no-sync rl @ "$config" --run.name "$run_name" --no-dashboard "${run_mode[@]}" "${ports[@]}" "$@"
