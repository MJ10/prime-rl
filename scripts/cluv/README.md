# Running on clusters with cluv

[cluv](https://github.com/mila-iqia/cluv) syncs this repository to Slurm clusters over SSH and
submits jobs there. Its config is the `[tool.cluv]` section of `pyproject.toml`.

`scripts/cluv/rl.sh` runs one single-node `rl` job per Slurm allocation: trainer, inference and
orchestrator share the node, so the config needs a `[deployment]` that fits it and no `[slurm]`
block.

## Setup

Install cluv as a tool, outside the project environment:

```bash
uv tool install cluster-uv
```

Your `~/.ssh/config` needs an entry for each cluster in `[tool.cluv.clusters]` (for Mila,
`milatools init` writes one).

cluv clones the repository on the cluster but does not initialize submodules, and its `uv sync`
needs `deps/verifiers` and the other submodules. Clone it once per cluster with submodules, and
set `submodule.recurse` so that the `git checkout`/`git pull` that cluv runs on every sync keeps
them current:

```bash
ssh mila 'git clone --recurse-submodules git@github.com:MJ10/prime-rl.git ~/repos/prime-rl && git -C ~/repos/prime-rl config submodule.recurse true'
```

Clone the remote your branch tracks, since cluv fetches and pushes through it. The clone path must
match `project_dir` (`$SCRATCH/repos/prime-rl` on `fir` and `trillium-gpu`). The
cluster needs a GitHub SSH key for the `git@github.com:` submodule URLs.

Then connect and sync:

```bash
cluv login mila
cluv sync mila
```

## Clusters

| Cluster | GPUs per node | Compute-node internet | Notes |
|---|---|---|---|
| `mila` | 4x A100-80GB (`80gb` constraint) | yes | `long` partition |
| `narval` | 4x A100-40GB | no | |
| `rorqual` | 4x H100-80GB | no | |
| `fir` | 4x H100-80GB | yes | requires a typed request (`h100:4`) |
| `nibi` | 8x H100-80GB | yes | requires a typed request (`h100:4`) |
| `trillium-gpu` | 4x H100-80GB | no | project on `$SCRATCH`, CPUs and memory fixed per GPU, 24h max |
| `tamia` | 4x H100-80GB | no | |

The `account` values are the bengioy group's allocations (`rrg-bengioy-ad`, `def-bengioy`,
`aip-bengioy` on tamia); replace them with your own. A list of `sbatch_args` submits one job per allocation and keeps
the first to start.

### Clusters without internet on compute nodes

`[tool.cluv.env]` sets `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline` for every
cluster that doesn't override them. On those clusters the job can't build its environment or
download the model, so prepare them on a login node after syncing and before submitting:

```bash
cluv sync narval
ssh narval 'cd ~/repos/prime-rl && GIT_COMMIT=$(git rev-parse HEAD) bash scripts/cluv/rl.sh --prepare examples/glyphbench/single_node.toml'
cluv submit narval -- examples/glyphbench/single_node.toml
```

`--prepare` builds the same per-commit snapshot that jobs use, then downloads the config's
`model.name` into `$HF_HOME`. A job whose snapshot isn't prepared fails right away. Upload offline
W&B runs from a login node with `wandb sync`.

## Submitting

cluv pushes the current branch to its remote and only submits from a clean working tree.

```bash
cluv submit mila -- examples/glyphbench/single_node.toml
```

Arguments after `--` are the config followed by `rl` overrides:

```bash
cluv submit mila -- examples/glyphbench/single_node.toml --max-steps 20 --trainer.optim.lr 2e-6
```

Arguments before `--` are `sbatch` flags and override the `#SBATCH` header of `rl.sh` and
`[tool.cluv.sbatch_args]`:

```bash
cluv submit nibi --gpus-per-node=h100:8 --time=3-00:00:00 -- examples/glyphbench/single_node.toml
```

`cluv submit first -- <config>` submits to every connected cluster and keeps whichever job starts
first; prepare the offline clusters beforehand.

Check jobs with `cluv status`. Slurm logs land in `$SCRATCH/logs/prime-rl/<cluster>_<job id>/`
(and in `logs/` locally after `cluv sync`).

## What a job does

1. Clones the submitted commit (`$GIT_COMMIT`) into `$SCRATCH/prime-rl/snapshots/<commit>`, with
   submodules cloned from the cluster checkout, and builds its environment with
   `uv sync --all-extras --locked`. Later jobs on the same commit reuse the snapshot, and later
   syncs never change the code under a queued or running job.
2. Runs `uv run rl @ <config> --run.name <config name>-<job id> --no-dashboard ...` from the
   snapshot, writing the run to `$SCRATCH/prime-rl/outputs/<config name>-<job id>`.
3. On a requeue (`requeue = true` in `[tool.cluv.sbatch_args]`), the same job id finds its run
   directory and passes `--resume` when it has a checkpoint, otherwise `--clean`. Set `[ckpt]` in
   the config for preemptible partitions.

The job sets these when they are unset: `UV_CACHE_DIR=$SCRATCH/.cache/uv`,
`HF_HOME=$SCRATCH/.cache/huggingface`, `PRL_OUTPUT_DIR=$SCRATCH/prime-rl/outputs`. Override them,
or add `WANDB_API_KEY`, through `env` in `[tool.cluv]` or `[tool.cluv.clusters.<name>]`.

Snapshots are not deleted automatically. Remove old ones from
`$SCRATCH/prime-rl/snapshots/` when no job uses them.

## Adding a cluster

Add a `[tool.cluv.clusters.<ssh host>]` section with its account, CPUs and memory (the job
script's header only requests GPUs and time). Override `env` with `UV_OFFLINE = "0"`,
`HF_HUB_OFFLINE = "0"` and `WANDB_MODE = "online"` if its compute nodes have internet.
