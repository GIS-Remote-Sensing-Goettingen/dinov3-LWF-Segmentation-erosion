"""Launch and manage one Slurm worker per fixed-size inference batch.

Examples:
    >>> isinstance(REPO_ROOT.name, str)
    True
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from segedge.core.config_loader import get_loaded_config_path  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "batches"
DEFAULT_TEMPLATE = REPO_ROOT / "silver_set.sh"
logger = logging.getLogger(__name__)
SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")


def _load_module(name: str, path: Path):
    """Load a Python module from a file path.

    Examples:
        >>> callable(_load_module)
        True
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_inference_tiles_for_batches(
    *, batch_root: Path
) -> tuple[list[str], str | None, str]:
    """Resolve and prefilter the inference tiles once for batch launch.

    Examples:
        >>> callable(resolve_inference_tiles_for_batches)
        True
    """
    module = _load_module(
        "build_inference_shards_module",
        REPO_ROOT / "deployment" / "build_inference_shards.py",
    )
    return module._resolve_inference_tiles_for_shards(shard_root=batch_root)


def merge_batch_unions(*args, **kwargs):
    """Load and dispatch to the merge helper lazily.

    Examples:
        >>> callable(merge_batch_unions)
        True
    """
    module = _load_module(
        "merge_shard_unions_module",
        REPO_ROOT / "deployment" / "merge_shard_unions.py",
    )
    return module.merge_shard_unions(*args, **kwargs)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Examples:
        >>> callable(_load_yaml)
        True
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a YAML mapping to disk.

    Examples:
        >>> callable(_write_yaml)
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a JSON file or return a default mapping.

    Examples:
        >>> _load_json(Path('/tmp/does-not-exist.json'), {'a': 1})['a']
        1
    """
    if not path.exists():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable pretty format.

    Examples:
        >>> callable(_write_json)
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strip_sbatch_overrides(sbatch_lines: list[str]) -> list[str]:
    """Drop scheduler directives that the renderer owns explicitly.

    Examples:
        >>> _strip_sbatch_overrides(['#SBATCH --job-name=x', '#SBATCH --mem=1G'])
        ['#SBATCH --mem=1G']
    """
    override_prefixes = (
        "#SBATCH --job-name=",
        "#SBATCH --output=",
        "#SBATCH --error=",
        "#SBATCH --array=",
    )
    return [
        line
        for line in sbatch_lines
        if not any(line.startswith(prefix) for prefix in override_prefixes)
    ]


def _is_template_main_invocation(line: str) -> bool:
    """Return whether a template line launches the main pipeline directly.

    Examples:
        >>> _is_template_main_invocation("python -u ./main.py --config cfg.yml")
        True
    """
    return line.strip().startswith("python -u ./main.py")


def _is_hardcoded_repo_cd(line: str) -> bool:
    """Return whether a template line hard-codes a checkout-specific `cd`.

    Examples:
        >>> _is_hardcoded_repo_cd(" cd /user/example/dinov3-LWF-Segmentation-erosion")
        True
    """
    stripped = line.strip()
    return stripped.startswith("cd ") and REPO_ROOT.name in stripped


def _repo_root_setup_lines() -> list[str]:
    """Return the shell stanza that validates and enters the repo root.

    Examples:
        >>> any("REPO_ROOT=" in line for line in _repo_root_setup_lines())
        True
    """
    return [
        f"REPO_ROOT={shlex.quote(str(REPO_ROOT))}",
        'if [ ! -f "${REPO_ROOT}/main.py" ]; then',
        '  echo "missing repo root main.py: ${REPO_ROOT}" >&2',
        "  exit 1",
        "fi",
        'cd "${REPO_ROOT}"',
    ]


def _render_slurm_script(
    *,
    template_path: Path,
    job_name: str,
    stdout_path: Path,
    stderr_path: Path,
    command: str,
) -> str:
    """Render a derived Slurm script from a read-only template.

    Examples:
        >>> callable(_render_slurm_script)
        True
    """
    lines = template_path.read_text(encoding="utf-8").splitlines()
    shebang = lines[0] if lines and lines[0].startswith("#!") else "#!/usr/bin/env bash"
    sbatch_lines: list[str] = []
    body_start = 1
    for idx, line in enumerate(lines[1:], start=1):
        if line.startswith("#SBATCH"):
            sbatch_lines.append(line)
            body_start = idx + 1
            continue
        if not line.strip():
            body_start = idx + 1
            continue
        body_start = idx
        break
    body_lines = [
        line
        for line in lines[body_start:]
        if not _is_template_main_invocation(line) and not _is_hardcoded_repo_cd(line)
    ]
    body_sections = [
        "\n".join(body_lines).rstrip(),
        "\n".join(_repo_root_setup_lines()),
        command,
    ]
    body = "\n\n".join(section for section in body_sections if section).rstrip()
    script_lines = [
        shebang,
        *_strip_sbatch_overrides(sbatch_lines),
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
        "",
        body.rstrip(),
        "",
    ]
    return "\n".join(script_lines)


def _write_executable(path: Path, content: str) -> Path:
    """Write an executable helper script.

    Examples:
        >>> callable(_write_executable)
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _chunk_tiles(tiles: list[str], batch_size: int) -> list[list[str]]:
    """Split tiles into contiguous fixed-size chunks.

    Examples:
        >>> _chunk_tiles(['a', 'b', 'c'], 2)
        [['a', 'b'], ['c']]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [tiles[idx : idx + batch_size] for idx in range(0, len(tiles), batch_size)]


def build_inference_batches(
    *,
    batch_size: int,
    output_dir: Path,
    job_name: str,
) -> tuple[Path, list[Path]]:
    """Resolve filtered tiles once and write deterministic batch tile lists.

    Examples:
        >>> callable(build_inference_batches)
        True
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    batch_root = output_dir / job_name
    batch_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "build batches: job=%s batch_size=%s output=%s",
        job_name,
        batch_size,
        batch_root,
    )
    tiles, inference_dir, inference_glob = resolve_inference_tiles_for_batches(
        batch_root=batch_root
    )
    logger.info("build batches: filtered tiles=%s", len(tiles))
    batches = _chunk_tiles(tiles, batch_size)
    batch_files: list[Path] = []
    for idx, batch_tiles in enumerate(batches):
        batch_path = batch_root / f"tiles_batch_{idx:03d}.txt"
        batch_path.write_text(
            "\n".join(batch_tiles) + ("\n" if batch_tiles else ""),
            encoding="utf-8",
        )
        batch_files.append(batch_path)
        logger.info(
            "build batches: wrote batch %03d with %s tiles -> %s",
            idx,
            len(batch_tiles),
            batch_path,
        )
    manifest = {
        "assignment_method": "contiguous_chunks",
        "batch_size": batch_size,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inference_dir": inference_dir,
        "inference_glob": inference_glob,
        "job_name": job_name,
        "num_batches": len(batch_files),
        "total_tiles": len(tiles),
        "tiles_file_paths": [str(path) for path in batch_files],
        "tiles_per_batch": [len(batch_tiles) for batch_tiles in batches],
    }
    _write_json(batch_root / "manifest.json", manifest)
    return batch_root, batch_files


def _load_tiles(path: Path) -> list[str]:
    """Load non-empty tile paths from a batch file.

    Examples:
        >>> callable(_load_tiles)
        True
    """
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _count_processed_tiles(run_dir: Path) -> int:
    """Count unique completed tiles in a batch run directory.

    Examples:
        >>> callable(_count_processed_tiles)
        True
    """
    processed_log = run_dir / "processed_tiles.jsonl"
    if not processed_log.exists():
        return 0
    done_tiles: set[str] = set()
    for line in processed_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "done" and payload.get("tile_path"):
            done_tiles.add(str(payload["tile_path"]))
    return len(done_tiles)


def _processed_tiles_path(run_dir: Path) -> Path:
    """Return the per-run processed-tiles log path.

    Examples:
        >>> _processed_tiles_path(Path('/tmp/run')).name
        'processed_tiles.jsonl'
    """
    return run_dir / "processed_tiles.jsonl"


def _run_log_path(run_dir: Path) -> Path:
    """Return the per-run plain-text log path.

    Examples:
        >>> _run_log_path(Path('/tmp/run')).name
        'run.log'
    """
    return run_dir / "run.log"


def _expected_run_dir(orchestration_root: Path, batch_idx: int) -> Path:
    """Return the fixed run directory for one batch.

    Examples:
        >>> _expected_run_dir(Path('/tmp/root'), 2).name
        'batch_002'
    """
    return orchestration_root / "runs" / f"batch_{batch_idx:03d}"


def _worker_log_path(
    orchestration_root: Path,
    job_id: str | None,
    batch_idx: int,
    suffix: str,
) -> str | None:
    """Return the concrete worker log path for one batch/job pair.

    Examples:
        >>> _worker_log_path(Path('/tmp/root'), '123', 2, 'out').endswith('worker_batch_002_123.out')
        True
    """
    if job_id in (None, ""):
        return None
    return str(
        orchestration_root / "slurm" / f"worker_batch_{batch_idx:03d}_{job_id}.{suffix}"
    )


def _tail_log_excerpt(path: Path, *, line_limit: int = 20) -> str | None:
    """Return a short tail excerpt from a text log, preferring error-like lines.

    Examples:
        >>> _tail_log_excerpt(Path('/tmp/does-not-exist.log')) is None
        True
    """
    if not path.exists():
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    markers = (
        "ERROR",
        "CRITICAL",
        "Traceback",
        "Exception",
        "No such file",
        "missing ",
    )
    for line in reversed(lines[-line_limit:]):
        if any(marker in line for marker in markers):
            return line[:400]
    return lines[-1][:400]


def _update_batch_diagnostics(
    *,
    batch: dict[str, Any],
    orchestration_root: Path,
) -> None:
    """Refresh one batch's file paths and last-known failure hint.

    Examples:
        >>> batch = {'batch_id': 0, 'run_dir': '/tmp/run', 'done_tiles': 0, 'expected_tiles': 1}
        >>> _update_batch_diagnostics(batch=batch, orchestration_root=Path('/tmp/root'))
        >>> batch['processed_tiles_path'].endswith('processed_tiles.jsonl')
        True
    """
    run_dir = Path(batch["run_dir"])
    batch_idx = int(batch["batch_id"])
    processed_path = _processed_tiles_path(run_dir)
    run_log_path = _run_log_path(run_dir)
    batch["processed_tiles_path"] = str(processed_path)
    batch["run_log_path"] = str(run_log_path)
    batch["run_dir_exists"] = run_dir.exists()
    batch.setdefault("worker_stdout_path", None)
    batch.setdefault("worker_stderr_path", None)
    batch.setdefault("last_worker_job_id", None)
    last_job_id = batch.get("last_worker_job_id")
    if last_job_id and not batch.get("worker_stdout_path"):
        batch["worker_stdout_path"] = _worker_log_path(
            orchestration_root,
            str(last_job_id),
            batch_idx,
            "out",
        )
    if last_job_id and not batch.get("worker_stderr_path"):
        batch["worker_stderr_path"] = _worker_log_path(
            orchestration_root,
            str(last_job_id),
            batch_idx,
            "err",
        )

    worker_stdout = (
        Path(batch["worker_stdout_path"]) if batch.get("worker_stdout_path") else None
    )
    worker_stderr = (
        Path(batch["worker_stderr_path"]) if batch.get("worker_stderr_path") else None
    )
    worker_logs_present = any(
        path is not None and path.exists() for path in (worker_stdout, worker_stderr)
    )
    batch["worker_logs_present"] = worker_logs_present

    if int(batch.get("done_tiles", 0)) >= int(batch.get("expected_tiles", 0)):
        batch["last_failure_source"] = None
        batch["last_failure_reason"] = None
        return

    candidates = [
        ("worker_stderr", worker_stderr),
        ("worker_stdout", worker_stdout),
        ("run_log", run_log_path),
    ]
    for source, path in candidates:
        if path is None:
            continue
        excerpt = _tail_log_excerpt(path)
        if excerpt:
            batch["last_failure_source"] = source
            batch["last_failure_reason"] = excerpt
            return

    batch["last_failure_source"] = "status"
    batch["last_failure_reason"] = (
        "run_dir_missing" if not run_dir.exists() else "no_worker_logs_found"
    )


def _apply_worker_submission_metadata(
    *,
    batch: dict[str, Any],
    orchestration_root: Path,
    worker_job_id: str | None,
) -> None:
    """Attach worker-job metadata to one batch status entry.

    Examples:
        >>> batch = {'batch_id': 0}
        >>> _apply_worker_submission_metadata(
        ...     batch=batch,
        ...     orchestration_root=Path('/tmp/root'),
        ...     worker_job_id='7',
        ... )
        >>> batch['last_worker_job_id']
        '7'
    """
    batch_idx = int(batch["batch_id"])
    batch["last_worker_job_id"] = worker_job_id
    batch["worker_stdout_path"] = _worker_log_path(
        orchestration_root,
        worker_job_id,
        batch_idx,
        "out",
    )
    batch["worker_stderr_path"] = _worker_log_path(
        orchestration_root,
        worker_job_id,
        batch_idx,
        "err",
    )
    batch["worker_logs_present"] = False
    batch["last_failure_source"] = None
    batch["last_failure_reason"] = None


def _write_batch_configs(
    *,
    orchestration_root: Path,
    batch_files: list[Path],
    base_config_path: Path,
) -> list[Path]:
    """Write one explicit config copy per batch.

    Examples:
        >>> callable(_write_batch_configs)
        True
    """
    base_cfg = _load_yaml(base_config_path)
    config_paths: list[Path] = []
    for idx, batch_file in enumerate(batch_files):
        batch_cfg = json.loads(json.dumps(base_cfg))
        batch_cfg.setdefault("io", {}).setdefault("inference", {})["tiles_file"] = str(
            batch_file
        )
        batch_cfg["io"]["inference"]["tiles_dir"] = None
        batch_cfg["io"]["inference"]["tiles"] = []
        batch_cfg.setdefault("runtime", {})
        batch_cfg["runtime"]["run_dir"] = str(
            _expected_run_dir(orchestration_root, idx)
        )
        batch_cfg["runtime"]["resume_run"] = True
        batch_cfg["runtime"]["resume_run_dir"] = str(
            _expected_run_dir(orchestration_root, idx)
        )
        config_path = orchestration_root / "configs" / f"batch_{idx:03d}.yml"
        _write_yaml(config_path, batch_cfg)
        config_paths.append(config_path)
    return config_paths


def _initial_status(
    *,
    orchestration_root: Path,
    batch_files: list[Path],
    worker_scripts: list[Path],
    max_retries: int,
    batch_size: int,
) -> dict[str, Any]:
    """Build initial orchestration status tracking.

    Examples:
        >>> callable(_initial_status)
        True
    """
    batches = []
    for idx, (batch_file, worker_script) in enumerate(zip(batch_files, worker_scripts)):
        run_dir = _expected_run_dir(orchestration_root, idx)
        batches.append(
            {
                "batch_id": idx,
                "tiles_file": str(batch_file),
                "worker_script": str(worker_script),
                "run_dir": str(run_dir),
                "processed_tiles_path": str(_processed_tiles_path(run_dir)),
                "run_log_path": str(_run_log_path(run_dir)),
                "expected_tiles": len(_load_tiles(batch_file)),
                "done_tiles": 0,
                "retry_count": 0,
                "status": "pending",
                "last_worker_job_id": None,
                "worker_stdout_path": None,
                "worker_stderr_path": None,
                "worker_logs_present": False,
                "run_dir_exists": False,
                "last_failure_source": None,
                "last_failure_reason": None,
            }
        )
    return {
        "batch_size": batch_size,
        "batches": batches,
        "controller_job_ids": [],
        "max_retries": max_retries,
        "state": "submitted",
        "worker_job_ids": [],
    }


def _refresh_status(orchestration_root: Path) -> dict[str, Any]:
    """Refresh batch completion counts from processed logs.

    Examples:
        >>> callable(_refresh_status)
        True
    """
    status_path = orchestration_root / "status.json"
    status = _load_json(status_path)
    if not status:
        raise ValueError(f"missing status file: {status_path}")
    for batch in status.get("batches", []):
        run_dir = Path(batch["run_dir"])
        batch["done_tiles"] = _count_processed_tiles(run_dir)
        batch["status"] = (
            "complete"
            if batch["done_tiles"] >= int(batch["expected_tiles"])
            else "incomplete"
        )
        _update_batch_diagnostics(batch=batch, orchestration_root=orchestration_root)
    _write_json(status_path, status)
    return status


def _incomplete_batch_payload(batch: dict[str, Any]) -> dict[str, Any]:
    """Build the failure-oriented summary for one incomplete batch.

    Examples:
        >>> payload = _incomplete_batch_payload({'batch_id': 1, 'done_tiles': 0, 'expected_tiles': 2})
        >>> payload['batch_id']
        1
    """
    return {
        "batch_id": batch["batch_id"],
        "done_tiles": batch["done_tiles"],
        "expected_tiles": batch["expected_tiles"],
        "run_dir": batch.get("run_dir"),
        "processed_tiles_path": batch.get("processed_tiles_path"),
        "run_log_path": batch.get("run_log_path"),
        "worker_stdout_path": batch.get("worker_stdout_path"),
        "worker_stderr_path": batch.get("worker_stderr_path"),
        "last_worker_job_id": batch.get("last_worker_job_id"),
        "run_dir_exists": batch.get("run_dir_exists"),
        "worker_logs_present": batch.get("worker_logs_present"),
        "last_failure_source": batch.get("last_failure_source"),
        "last_failure_reason": batch.get("last_failure_reason"),
    }


def _submit_sbatch(
    *,
    script_path: Path,
    extra_args: list[str] | None = None,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit an sbatch command or return a dry-run record.

    Examples:
        >>> callable(_submit_sbatch)
        True
    """
    command = ["sbatch", *(extra_args or []), str(script_path)]
    if dry_run:
        return {"job_id": None, "command": command}
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    match = SBATCH_ID_RE.search(proc.stdout.strip())
    if not match:
        raise RuntimeError(f"unable to parse sbatch output: {proc.stdout!r}")
    return {"job_id": match.group(1), "command": command, "stdout": proc.stdout.strip()}


def _write_submission(orchestration_root: Path, payload: dict[str, Any]) -> None:
    """Persist Slurm submission metadata.

    Examples:
        >>> callable(_write_submission)
        True
    """
    _write_json(orchestration_root / "submission.json", payload)


def _load_existing_orchestration(
    orchestration_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load manifest and status for an existing orchestration root.

    Examples:
        >>> callable(_load_existing_orchestration)
        True
    """
    manifest = _load_json(orchestration_root / "manifest.json")
    status = _load_json(orchestration_root / "status.json")
    if not manifest:
        raise ValueError(
            f"missing manifest for existing orchestration root: {orchestration_root}"
        )
    if not status:
        raise ValueError(
            f"missing status for existing orchestration root: {orchestration_root}"
        )
    return manifest, status


def _worker_script_command(config_path: Path) -> str:
    """Build one batch worker command snippet.

    Examples:
        >>> '--config' in _worker_script_command(Path('/tmp/cfg.yml'))
        True
    """
    return "\n".join(
        [
            f"CONFIG_PATH={shlex.quote(str(config_path))}",
            'if [ ! -f "${CONFIG_PATH}" ]; then',
            '  echo "missing batch config: ${CONFIG_PATH}" >&2',
            "  exit 1",
            "fi",
            'python -u ./main.py --config "${CONFIG_PATH}"',
        ]
    )


def _controller_command(orchestration_root: Path) -> str:
    """Build a self-referential launcher command for the controller stage.

    Examples:
        >>> '--controller' in _controller_command(Path('/tmp/root'))
        True
    """
    script_path = REPO_ROOT / "deployment" / "launch_batched_inference.py"
    return (
        f'python -u "{script_path}" --controller '
        f'--orchestration-root "{orchestration_root}"'
    )


def _write_slurm_scripts(
    *,
    orchestration_root: Path,
    template_path: Path,
    job_name: str,
    config_paths: list[Path],
) -> tuple[list[Path], Path]:
    """Render worker and controller Slurm scripts.

    Examples:
        >>> callable(_write_slurm_scripts)
        True
    """
    slurm_dir = orchestration_root / "slurm"
    worker_scripts: list[Path] = []
    for idx, config_path in enumerate(config_paths):
        worker_scripts.append(
            _write_executable(
                slurm_dir / f"worker_batch_{idx:03d}.sh",
                _render_slurm_script(
                    template_path=template_path,
                    job_name=f"{job_name}_batch_{idx:03d}",
                    stdout_path=slurm_dir / f"worker_batch_{idx:03d}_%j.out",
                    stderr_path=slurm_dir / f"worker_batch_{idx:03d}_%j.err",
                    command=_worker_script_command(config_path),
                ),
            )
        )
    controller_script = _write_executable(
        slurm_dir / "controller.sh",
        _render_slurm_script(
            template_path=template_path,
            job_name=f"{job_name}_controller",
            stdout_path=slurm_dir / "controller_%j.out",
            stderr_path=slurm_dir / "controller_%j.err",
            command=_controller_command(orchestration_root),
        ),
    )
    return worker_scripts, controller_script


def _dependency_args(job_ids: list[str | None]) -> list[str]:
    """Build one afterany dependency argument covering all given job ids.

    Examples:
        >>> _dependency_args(['1', '2'])
        ['--dependency', 'afterany:1:2']
    """
    valid_ids = [str(job_id) for job_id in job_ids if job_id not in (None, "")]
    if not valid_ids:
        return []
    return ["--dependency", f"afterany:{':'.join(valid_ids)}"]


def launch_batched_inference(
    *,
    job_name: str,
    batch_size: int,
    template_path: Path,
    output_root: Path,
    max_retries: int,
    dry_run: bool,
) -> Path:
    """Build batch artifacts and submit one worker per batch.

    Examples:
        >>> callable(launch_batched_inference)
        True
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    logger.info(
        "launch batches: job=%s batch_size=%s template=%s output_root=%s dry_run=%s max_retries=%s",
        job_name,
        batch_size,
        template_path,
        output_root,
        dry_run,
        max_retries,
    )
    orchestration_root, batch_files = build_inference_batches(
        batch_size=batch_size,
        output_dir=output_root,
        job_name=job_name,
    )
    logger.info(
        "launch batches: batch planning complete root=%s batch_count=%s",
        orchestration_root,
        len(batch_files),
    )
    base_config_path = Path(get_loaded_config_path())
    config_paths = _write_batch_configs(
        orchestration_root=orchestration_root,
        batch_files=batch_files,
        base_config_path=base_config_path,
    )
    worker_scripts, controller_script = _write_slurm_scripts(
        orchestration_root=orchestration_root,
        template_path=template_path,
        job_name=job_name,
        config_paths=config_paths,
    )
    manifest = _load_json(orchestration_root / "manifest.json")
    manifest.update(
        {
            "base_config_path": str(base_config_path),
            "config_paths": [str(path) for path in config_paths],
            "controller_script": str(controller_script),
            "repo_root": str(REPO_ROOT),
            "run_dirs": [
                str(_expected_run_dir(orchestration_root, idx))
                for idx in range(len(batch_files))
            ],
            "slurm_scripts": {
                "controller": str(controller_script),
                "workers": [str(path) for path in worker_scripts],
            },
            "worker_log_dir": str(orchestration_root / "slurm"),
        }
    )
    _write_json(orchestration_root / "manifest.json", manifest)
    status = _initial_status(
        orchestration_root=orchestration_root,
        batch_files=batch_files,
        worker_scripts=worker_scripts,
        max_retries=max_retries,
        batch_size=batch_size,
    )

    worker_submissions: list[dict[str, Any]] = []
    worker_job_ids: list[str | None] = []
    for batch in status["batches"]:
        submission = _submit_sbatch(
            script_path=Path(batch["worker_script"]),
            extra_args=None,
            dry_run=dry_run,
        )
        worker_submissions.append(
            {
                "batch_id": batch["batch_id"],
                "submission": submission,
            }
        )
        worker_job_ids.append(submission["job_id"])
        status["worker_job_ids"].append(submission["job_id"])
        _apply_worker_submission_metadata(
            batch=batch,
            orchestration_root=orchestration_root,
            worker_job_id=submission["job_id"],
        )

    controller_submission = _submit_sbatch(
        script_path=controller_script,
        extra_args=_dependency_args(worker_job_ids),
        dry_run=dry_run,
    )
    status["controller_job_ids"].append(controller_submission["job_id"])
    _write_json(orchestration_root / "status.json", status)
    _write_submission(
        orchestration_root,
        {
            "controller_submission": controller_submission,
            "dry_run": dry_run,
            "worker_submissions": worker_submissions,
        },
    )
    return orchestration_root


def submit_controller_only(
    *,
    orchestration_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit only the existing controller Slurm script for a batch run.

    Examples:
        >>> callable(submit_controller_only)
        True
    """
    logger.info(
        "submit controller only: root=%s dry_run=%s", orchestration_root, dry_run
    )
    manifest, status = _load_existing_orchestration(orchestration_root)
    controller_script = manifest.get("controller_script")
    if not controller_script:
        raise ValueError(
            f"missing controller_script in manifest: {orchestration_root / 'manifest.json'}"
        )
    submission = _submit_sbatch(
        script_path=Path(controller_script),
        extra_args=None,
        dry_run=dry_run,
    )
    status.setdefault("controller_job_ids", []).append(submission["job_id"])
    status["state"] = "controller_submitted"
    _write_json(orchestration_root / "status.json", status)
    _write_json(
        orchestration_root / "submission.json",
        {
            **_load_json(orchestration_root / "submission.json"),
            "manual_controller_submission": submission,
        },
    )
    return submission


def run_controller(*, orchestration_root: Path, dry_run: bool) -> None:
    """Retry incomplete batches or merge completed batch outputs.

    Examples:
        >>> callable(run_controller)
        True
    """
    logger.info("controller: start root=%s dry_run=%s", orchestration_root, dry_run)
    manifest = _load_json(orchestration_root / "manifest.json")
    status = _refresh_status(orchestration_root)
    incomplete = [
        batch
        for batch in status["batches"]
        if batch["done_tiles"] < batch["expected_tiles"]
    ]
    for batch in status["batches"]:
        logger.info(
            (
                "controller: batch=%03d done=%s expected=%s retries=%s status=%s "
                "run_dir_exists=%s worker_logs_present=%s "
                "failure_source=%s failure_reason=%s"
            ),
            int(batch["batch_id"]),
            int(batch["done_tiles"]),
            int(batch["expected_tiles"]),
            int(batch["retry_count"]),
            batch["status"],
            bool(batch.get("run_dir_exists")),
            bool(batch.get("worker_logs_present")),
            batch.get("last_failure_source"),
            batch.get("last_failure_reason"),
        )

    if not incomplete:
        merged_dir = orchestration_root / "merged"
        merged_paths = merge_batch_unions(
            shard_run_dirs=[str(Path(batch["run_dir"])) for batch in status["batches"]],
            output_dir=str(merged_dir),
        )
        status["state"] = "complete"
        _write_json(orchestration_root / "status.json", status)
        _write_json(
            orchestration_root / "final_status.json",
            {
                "status": "success",
                "merged_dir": str(merged_dir),
                "merged_paths": [str(path) for path in merged_paths],
            },
        )
        logger.info("controller: merged unions -> %s", merged_dir)
        return

    retryable = [
        batch
        for batch in incomplete
        if int(batch["retry_count"]) < int(status["max_retries"])
    ]
    if not retryable:
        status["state"] = "failed_incomplete"
        _write_json(orchestration_root / "status.json", status)
        _write_json(
            orchestration_root / "final_status.json",
            {
                "status": "failed_incomplete",
                "incomplete_batches": [
                    _incomplete_batch_payload(batch) for batch in incomplete
                ],
            },
        )
        logger.error(
            "controller: retry budget exhausted incomplete=%s",
            [int(batch["batch_id"]) for batch in incomplete],
        )
        raise SystemExit("incomplete batches remain and retry limit is exhausted")

    logger.info(
        "controller: retrying batches=%s",
        [int(batch["batch_id"]) for batch in retryable],
    )
    retry_submissions: list[dict[str, Any]] = []
    retry_job_ids: list[str | None] = []
    for batch in status["batches"]:
        if batch not in retryable:
            continue
        batch["retry_count"] = int(batch["retry_count"]) + 1
        batch["status"] = "retrying"
        submission = _submit_sbatch(
            script_path=Path(batch["worker_script"]),
            extra_args=None,
            dry_run=dry_run,
        )
        retry_submissions.append(
            {
                "batch_id": batch["batch_id"],
                "submission": submission,
            }
        )
        retry_job_ids.append(submission["job_id"])
        status["worker_job_ids"].append(submission["job_id"])
        _apply_worker_submission_metadata(
            batch=batch,
            orchestration_root=orchestration_root,
            worker_job_id=submission["job_id"],
        )

    controller_submission = _submit_sbatch(
        script_path=Path(manifest["controller_script"]),
        extra_args=_dependency_args(retry_job_ids),
        dry_run=dry_run,
    )
    status["controller_job_ids"].append(controller_submission["job_id"])
    status["state"] = "retry_submitted"
    _write_json(orchestration_root / "status.json", status)
    _write_json(
        orchestration_root / "submission.json",
        {
            **_load_json(orchestration_root / "submission.json"),
            "last_retry_controller_submission": controller_submission,
            "last_retry_worker_submissions": retry_submissions,
        },
    )


def main() -> None:
    """CLI entrypoint for simple batch-based Slurm inference launches."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-name",
        help="Name for the batch orchestration root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum number of tiles per worker batch.",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Read-only Slurm template file used to render worker/controller scripts.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where batch manifests, configs, scripts, and merged outputs live.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry submissions for incomplete batch runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render batch artifacts without calling sbatch.",
    )
    parser.add_argument(
        "--controller",
        action="store_true",
        help="Internal mode: inspect batch progress, retry incomplete batches, and merge when complete.",
    )
    parser.add_argument(
        "--submit-controller",
        action="store_true",
        help="Submit only the existing controller Slurm script for an orchestration root.",
    )
    parser.add_argument(
        "--orchestration-root",
        help="Existing orchestration root for controller-oriented modes.",
    )
    args = parser.parse_args()

    if args.controller:
        if not args.orchestration_root:
            raise ValueError("--orchestration-root is required with --controller")
        run_controller(
            orchestration_root=Path(args.orchestration_root),
            dry_run=bool(args.dry_run),
        )
        return

    if args.submit_controller:
        if not args.orchestration_root:
            raise ValueError(
                "--orchestration-root is required with --submit-controller"
            )
        submit_controller_only(
            orchestration_root=Path(args.orchestration_root),
            dry_run=bool(args.dry_run),
        )
        return

    if not args.job_name:
        raise ValueError("--job-name is required for launch mode")
    launch_batched_inference(
        job_name=args.job_name,
        batch_size=int(args.batch_size),
        template_path=Path(args.template),
        output_root=Path(args.output_root),
        max_retries=int(args.max_retries),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
