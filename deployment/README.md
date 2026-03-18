# Deployment

This folder contains the operational entrypoints for large inference campaigns.

The intended use case is:
- split one large inference folder into deterministic tile batches
- run one isolated Slurm task per batch
- retry incomplete batches safely
- verify completion
- merge the final union rasters

The supported orchestration style is:
- `launch_batched_inference.py`: one ordinary Slurm job per fixed-size batch

## Scripts

### `launch_batched_inference.py`
Use this when you want the simpler operational model of one plain Slurm job per
batch and no array indexing.

What it does:
- resolves the inference tile set once
- applies the source-label filter once
- writes one `tiles_batch_XXX.txt` file per worker batch
- writes one batch-specific config file per worker batch
- renders one worker script per batch plus one controller script from `silver_set.sh`
- pins generated Slurm scripts to the current repository checkout instead of trusting a hard-coded template `cd`
- submits one ordinary `sbatch` worker job per batch
- submits one dependent controller job that retries incomplete batches and merges unions when all batches finish

Typical command:

```bash
python deployment/launch_batched_inference.py \
  --job-name folder1_batches \
  --batch-size 100 \
  --template silver_set.sh
```

Useful options:
- `--dry-run`: render all artifacts without calling `sbatch`
- `--max-retries`: maximum retry waves for incomplete batches
- `--output-root`: root directory for batch orchestration outputs
- `--submit-controller --orchestration-root <root>`: submit only the existing controller Slurm script for a previously created batch orchestration root

Internal mode used by the generated controller job:
- `--controller`

Do not run it manually unless you are debugging the orchestration state.

### `build_inference_shards.py`
This helper is kept for shared tile-resolution and prefilter logic that the
batch launcher reuses. It is not the primary deployment entrypoint anymore.

What it does:
- resolves the configured inference tiles
- applies the source-label filter once
- partitions tiles by deterministic round-robin assignment
- writes one `tiles_shard_XXX.txt` file per shard
- writes `manifest.json`

Example:

```bash
python deployment/build_inference_shards.py --shards 4 --job-name folder1_helper
```

### `merge_shard_unions.py`
Use this when you want to merge completed batch outputs manually.

It merges these union families:
- `raw`
- `crf`
- `shadow`
- `shadow_with_proposals`

Example:

```bash
python deployment/merge_shard_unions.py \
  --output-dir output/batches/folder1_batches/merged \
  output/batches/folder1_batches/runs/batch_000 \
  output/batches/folder1_batches/runs/batch_001 \
  output/batches/folder1_batches/runs/batch_002 \
  output/batches/folder1_batches/runs/batch_003
```

Merge semantics:
- source union rasters may cover different extents
- the merge output is a GeoTIFF mosaic on the shared campaign grid
- merge still fails if inputs disagree on CRS or pixel size

## Orchestration layout

For a job name like `folder1_batches`, the launcher writes under:

```text
output/batches/folder1_batches/
```

Important contents:
- `manifest.json`: static batch/config/script metadata
- `status.json`: evolving orchestration state plus per-batch diagnostics
- `submission.json`: submitted job ids and sbatch commands
- `final_status.json`: final verification result
- `configs/batch_XXX.yml`: generated batch-specific configs
- `slurm/`: rendered worker/controller scripts
- `runs/batch_XXX/`: fixed batch run directories
- `merged/`: final merged union rasters after successful verification

## Retry and resume behavior

Each batch runs in a fixed run directory:

```text
output/batches/<job_name>/runs/batch_XXX/
```

That matters because retries do not start from scratch. They reuse:
- `processed_tiles.jsonl`
- rolling union rasters
- rolling checkpoint state

So if a batch does not finish all its tiles, the controller resubmits only that
batch and the new worker resumes the same batch run instead of creating a new
`run_*` directory.

This is the safe model for this repository. The launcher does **not** run
duplicate workers on the same batch concurrently.

For debugging, each batch status record now also tracks:
- the current batch run dir and its `processed_tiles.jsonl` / `run.log`
- the latest known worker stdout/stderr paths under `slurm/worker_batch_<id>_<job>.out|err`
- whether the run dir or worker logs exist yet
- a short last-known failure hint pulled from worker stderr/stdout or `run.log`

## Verification model

The launcher does not block locally waiting for the full campaign to finish.

Instead:
1. one worker job is submitted per batch
2. the controller checks batch completion counts
3. incomplete batches are retried up to `--max-retries`
4. the final controller step checks that every batch completed all expected tiles
5. only then are merged union mosaics written

If any batch is still incomplete after the retry budget is exhausted:
- the workflow stops
- merge is not performed
- `final_status.json` records failure plus the last-known per-batch diagnostics

## Config behavior

Batch workers use generated config copies rather than mutating the repo root `config.yml`.

Each generated config overrides only the batch-specific fields:
- `io.inference.tiles_file`
- `runtime.run_dir`
- `runtime.resume_run`
- `runtime.resume_run_dir`

The pipeline is executed with:

```bash
python -u ./main.py --config <generated-config>
```

This keeps the worker runs explicit, reproducible, and isolated.

Generated worker scripts still inherit scheduler and environment setup from the
template, but the launcher now renders the repository root explicitly before
launching Python so a stale template checkout path cannot silently break every
batch.
