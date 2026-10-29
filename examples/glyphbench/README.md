# GlyphBench

RL on [GlyphBench](https://github.com/VmaxAI/glyphbench) text-rendered game environments with
Qwen3.5-4B, ported from GlyphBench's prime-rl recipes. The `glyphbench` extra installs its
Verifiers v1 taskset, env and harness:

```bash
uv sync --all-extras
```

## Configs

| Config | Deployment | Tasks |
|---|---|---|
| `single_node.toml` | one 4-GPU node: 2 inference, 2 trainer GPUs | `tasks/pilot.txt` (8 tasks), 32K context, 8 groups x 8 |
| `single_task.toml` | 2 nodes x 8 GPUs, prime-rl's SLURM launcher | Snake Easy, 1 group x 512 |
| `multitask.toml` | 4 nodes x 8 GPUs, prime-rl's SLURM launcher | `tasks/craftax100.txt`, 64K context, 16 groups x 64 |

Two overlays layer on `single_node.toml` with `rl @ single_node.toml @ <overlay>`:
`snake_easy.toml` trains and evaluates on Snake Easy alone, and `no_thinking.toml` turns off
Qwen3.5 thinking with a 512-token turn cap. Together they give a fast learning check (about a
minute per step on 4 A100s).

`tasks/hero100.txt` is the GlyphBench paper's 100-task mixture and `tasks/fixed_eval.txt` its fixed
50-task eval panel. Point a source's `env.taskset.task_file` at any manifest; paths are relative to
the repository root, so launch from there.

GlyphBench sources configure the `player` agent role (`env.player.*`), not `env.agent.*`.

## Running on a cluster with cluv

`single_node.toml` runs inside one allocation through `scripts/cluv/rl.sh`
(see [scripts/cluv/README.md](../../scripts/cluv/README.md)):

```bash
cluv submit mila -- examples/glyphbench/single_node.toml
cluv submit mila -- examples/glyphbench/single_node.toml --monitors.wandb.project glyphbench-rl
```

On an 8-GPU node, request the GPUs and split them:

```bash
cluv submit nibi --gpus-per-node=h100:8 -- examples/glyphbench/single_node.toml \
  --deployment.gpus-per-node 8 --deployment.num-infer-gpus 4 --deployment.num-train-gpus 4 \
  --inference.vllm.data-parallel-size 4 --inference.vllm.api-server-count 4
```

The multi-node configs use prime-rl's own SLURM launcher (`uv run rl @ <config>` from a login
node); set `[slurm] partition` and the node counts for your cluster first.

## Differences from GlyphBench's v0.9 recipes

- `max_staleness = 3` replaces `max_off_policy_steps = 3`: stale tokens are masked instead of
  whole stale episodes being dropped.
- The IPO KL term (`kl_tau`) no longer exists.
- Zero-advantage groups are always filtered; `constant_trainer_batch_size = false` keeps the
  recipes' no-backfill behaviour.
- `single_task.toml`'s selective activation checkpointing saves only the attention kernels,
  since `ac.targets` now lists operators to keep rather than modules to recompute.
- `optimization_dtype` and `reduce_dtype` are `bfloat16`, as in the recipes; prime-rl defaults to
  `float32`.
