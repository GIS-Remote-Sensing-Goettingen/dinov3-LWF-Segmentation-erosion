# Deployment

This folder contains the operational entrypoints for large inference campaigns.

The intended use case is:
- split one large inference folder into deterministic shards
- run one isolated Slurm task per shard
- retry incomplete shards safely
- verify completion
- merge the final union shapefiles

## Scripts

### `orchestrate_sharded_inference.py`
Use this as the main entrypoint for Slurm-backed inference orchestration.

What it does:
- resolves the inference tile set once
- applies the source-label filter once
- writes one shard file per worker
- writes one shard-specific config file per worker
- renders worker, watchdog, and verify/merge Slurm scripts from `silver_set.sh`
- pins generated Slurm scripts to the current repository checkout instead of trusting a hard-coded template `cd`
- submits one Slurm job array for shard workers
- submits a dependent watchdog job
- resubmits only incomplete shards
- runs final verification and merges unions only when all shards are complete

Typical command:

```bash
python deployment/orchestrate_sharded_inference.py \
  --job-name folder1_4way \
  --shards 4 \
  --template silver_set.sh
```

Useful options:
- `--dry-run`: render all artifacts without calling `sbatch`
- `--max-retries`: maximum retry waves for incomplete shards
- `--output-root`: root directory for shard orchestration outputs

Internal modes used by the generated Slurm jobs:
- `--watchdog`
- `--verify-merge`

Do not run those manually unless you are debugging the orchestration state.

### `build_inference_shards.py`
Use this when you only want the shard files and manifest without submitting jobs.

What it does:
- resolves the configured inference tiles
- applies the source-label filter once
- partitions tiles by deterministic round-robin assignment
- writes one `tiles_shard_XXX.txt` file per shard
- writes `manifest.json`

Example:

```bash
python deployment/build_inference_shards.py --shards 4 --job-name folder1_4way
```

### `merge_shard_unions.py`
Use this when you want to merge completed shard outputs manually.

It merges these union families:
- `raw`
- `crf`
- `shadow`
- `shadow_with_proposals`

Example:

```bash
python deployment/merge_shard_unions.py \
  --output-dir output/shards/folder1_4way_merged \
  output/shards/folder1_4way/runs/shard_000 \
  output/shards/folder1_4way/runs/shard_001 \
  output/shards/folder1_4way/runs/shard_002 \
  output/shards/folder1_4way/runs/shard_003
```

## Orchestration layout

For a job name like `folder1_4way`, the orchestrator writes under:

```text
output/shards/folder1_4way/
```

Important contents:
- `manifest.json`: static shard/config/script metadata
- `status.json`: evolving orchestration state plus per-shard diagnostics
- `submission.json`: submitted job ids and sbatch commands
- `final_status.json`: final verification result
- `configs/shard_XXX.yml`: generated shard-specific configs
- `slurm/`: rendered worker/watchdog/verify scripts
- `runs/shard_XXX/`: fixed shard run directories
- `merged/`: final merged unions after successful verification

## Retry and resume behavior

Each shard runs in a fixed run directory:

```text
output/shards/<job_name>/runs/shard_XXX/
```

That matters because retries do not start from scratch. They reuse:
- `processed_tiles.jsonl`
- rolling union shapefiles
- rolling checkpoint state

So if a shard does not finish all its tiles, the watchdog resubmits that shard index and the new worker resumes the same shard run instead of creating a new `run_*` directory.

This is the safe model for this repository. The orchestrator does **not** run duplicate workers on the same shard concurrently.

For debugging, each shard status record now also tracks:
- the current shard run dir and its `processed_tiles.jsonl` / `run.log`
- the latest known worker stdout/stderr paths under `slurm/worker_<job>_<shard>.out|err`
- whether the run dir or worker logs exist yet
- a short last-known failure hint pulled from worker stderr/stdout or `run.log`

## Verification model

The orchestrator does not block locally waiting for the full campaign to finish.

Instead:
1. the worker array runs
2. the watchdog checks shard completion counts
3. incomplete shards are retried up to `--max-retries`
4. the final verify/merge step checks that every shard completed all expected tiles
5. only then are merged unions written

If any shard is still incomplete after the retry budget is exhausted:
- the workflow stops
- merge is not performed
- `final_status.json` records failure plus the last-known per-shard diagnostics

## Config behavior

Shard workers use generated config copies rather than mutating the repo root `config.yml`.

Each generated config overrides only the shard-specific fields:
- `io.inference.tiles_file`
- `runtime.run_dir`
- `runtime.resume_run`
- `runtime.resume_run_dir`

The pipeline is executed with:

```bash
python -u ./main.py --config <generated-config>
```

This keeps the worker runs explicit, reproducible, and isolated.

Generated worker scripts still inherit scheduler and environment setup from the template, but the orchestrator now renders the repository root explicitly before launching Python so a stale template checkout path cannot silently break every shard.
