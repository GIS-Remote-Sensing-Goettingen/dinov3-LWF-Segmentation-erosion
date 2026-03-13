"""Launch and manage sharded inference on Slurm.

Examples:
    >>> isinstance(REPO_ROOT.name, str)
    True
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from segedge.core.config_loader import get_loaded_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "shards"
DEFAULT_TEMPLATE = REPO_ROOT / "silver_set.sh"


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


_BUILD_MODULE = _load_module(
    "build_inference_shards_module",
    REPO_ROOT / "deployment" / "build_inference_shards.py",
)
_MERGE_MODULE = _load_module(
    "merge_shard_unions_module",
    REPO_ROOT / "deployment" / "merge_shard_unions.py",
)
build_inference_shards = _BUILD_MODULE.build_inference_shards
merge_shard_unions = _MERGE_MODULE.merge_shard_unions


SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    body = "\n".join(lines[body_start:]).rstrip() + "\n"
    body = body.replace("python -u ./main.py", command, 1)
    if command not in body:
        body = body.rstrip() + "\n" + command + "\n"
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


def _load_tiles(path: Path) -> list[str]:
    """Load non-empty tile paths from a shard file.

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
    """Count unique completed tiles in a shard run directory.

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


def _expected_run_dir(orchestration_root: Path, shard_idx: int) -> Path:
    """Return the fixed run directory for one shard.

    Examples:
        >>> _expected_run_dir(Path('/tmp/root'), 2).name
        'shard_002'
    """
    return orchestration_root / "runs" / f"shard_{shard_idx:03d}"


def _write_shard_configs(
    *,
    orchestration_root: Path,
    shard_files: list[Path],
    base_config_path: Path,
) -> list[Path]:
    """Write one explicit config copy per shard.

    Examples:
        >>> callable(_write_shard_configs)
        True
    """
    base_cfg = _load_yaml(base_config_path)
    config_paths: list[Path] = []
    for idx, shard_file in enumerate(shard_files):
        shard_cfg = json.loads(json.dumps(base_cfg))
        shard_cfg.setdefault("io", {}).setdefault("inference", {})["tiles_file"] = str(
            shard_file
        )
        shard_cfg.setdefault("runtime", {})
        shard_cfg["runtime"]["run_dir"] = str(
            _expected_run_dir(orchestration_root, idx)
        )
        shard_cfg["runtime"]["resume_run"] = True
        shard_cfg["runtime"]["resume_run_dir"] = str(
            _expected_run_dir(orchestration_root, idx)
        )
        config_path = orchestration_root / "configs" / f"shard_{idx:03d}.yml"
        _write_yaml(config_path, shard_cfg)
        config_paths.append(config_path)
    return config_paths


def _initial_status(
    *,
    orchestration_root: Path,
    shard_files: list[Path],
    max_retries: int,
) -> dict[str, Any]:
    """Build initial orchestration status tracking.

    Examples:
        >>> callable(_initial_status)
        True
    """
    shards = []
    for idx, shard_file in enumerate(shard_files):
        shards.append(
            {
                "shard_id": idx,
                "tiles_file": str(shard_file),
                "run_dir": str(_expected_run_dir(orchestration_root, idx)),
                "expected_tiles": len(_load_tiles(shard_file)),
                "done_tiles": 0,
                "retry_count": 0,
                "status": "pending",
            }
        )
    return {
        "max_retries": max_retries,
        "shards": shards,
        "worker_job_ids": [],
        "watchdog_job_ids": [],
        "merge_job_id": None,
        "state": "submitted",
    }


def _refresh_status(orchestration_root: Path) -> dict[str, Any]:
    """Refresh shard completion counts from processed logs.

    Examples:
        >>> callable(_refresh_status)
        True
    """
    status_path = orchestration_root / "status.json"
    status = _load_json(status_path)
    if not status:
        raise ValueError(f"missing status file: {status_path}")
    for shard in status.get("shards", []):
        run_dir = Path(shard["run_dir"])
        shard["done_tiles"] = _count_processed_tiles(run_dir)
        shard["status"] = (
            "complete"
            if shard["done_tiles"] >= int(shard["expected_tiles"])
            else "incomplete"
        )
    _write_json(status_path, status)
    return status


def _array_spec(indices: list[int]) -> str:
    """Format an sbatch array specification.

    Examples:
        >>> _array_spec([2, 5, 8])
        '2,5,8'
    """
    if not indices:
        raise ValueError("indices must not be empty")
    return ",".join(str(idx) for idx in sorted(indices))


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


def _load_manifest(orchestration_root: Path) -> dict[str, Any]:
    """Load the orchestration manifest.

    Examples:
        >>> callable(_load_manifest)
        True
    """
    return _load_json(orchestration_root / "manifest.json")


def _write_submission(orchestration_root: Path, payload: dict[str, Any]) -> None:
    """Persist Slurm submission metadata.

    Examples:
        >>> callable(_write_submission)
        True
    """
    _write_json(orchestration_root / "submission.json", payload)


def _worker_script_command(orchestration_root: Path) -> str:
    """Build the shard worker command snippet.

    Examples:
        >>> 'SLURM_ARRAY_TASK_ID' in _worker_script_command(Path('/tmp/root'))
        True
    """
    config_dir = orchestration_root / "configs"
    return "\n".join(
        [
            'SHARD_IDX="$(printf \'%03d\' "${SLURM_ARRAY_TASK_ID}")"',
            f'CONFIG_PATH="{config_dir}/shard_${{SHARD_IDX}}.yml"',
            'if [ ! -f "${CONFIG_PATH}" ]; then',
            '  echo "missing shard config: ${CONFIG_PATH}" >&2',
            "  exit 1",
            "fi",
            'python -u ./main.py --config "${CONFIG_PATH}"',
        ]
    )


def _orchestrator_command(mode: str, orchestration_root: Path) -> str:
    """Build a self-referential launcher command for watchdog stages.

    Examples:
        >>> '--watchdog' in _orchestrator_command('--watchdog', Path('/tmp/root'))
        True
    """
    script_path = REPO_ROOT / "deployment" / "orchestrate_sharded_inference.py"
    return (
        f'python -u "{script_path}" {mode} '
        f'--orchestration-root "{orchestration_root}"'
    )


def _write_slurm_scripts(
    *,
    orchestration_root: Path,
    template_path: Path,
    job_name: str,
) -> dict[str, Path]:
    """Render worker, watchdog, and verify Slurm scripts.

    Examples:
        >>> callable(_write_slurm_scripts)
        True
    """
    slurm_dir = orchestration_root / "slurm"
    worker = _write_executable(
        slurm_dir / "worker_array.sh",
        _render_slurm_script(
            template_path=template_path,
            job_name=f"{job_name}_worker",
            stdout_path=slurm_dir / "worker_%A_%a.out",
            stderr_path=slurm_dir / "worker_%A_%a.err",
            command=_worker_script_command(orchestration_root),
        ),
    )
    watchdog = _write_executable(
        slurm_dir / "watchdog.sh",
        _render_slurm_script(
            template_path=template_path,
            job_name=f"{job_name}_watchdog",
            stdout_path=slurm_dir / "watchdog_%j.out",
            stderr_path=slurm_dir / "watchdog_%j.err",
            command=_orchestrator_command("--watchdog", orchestration_root),
        ),
    )
    verify_merge = _write_executable(
        slurm_dir / "verify_merge.sh",
        _render_slurm_script(
            template_path=template_path,
            job_name=f"{job_name}_merge",
            stdout_path=slurm_dir / "merge_%j.out",
            stderr_path=slurm_dir / "merge_%j.err",
            command=_orchestrator_command("--verify-merge", orchestration_root),
        ),
    )
    return {"worker": worker, "watchdog": watchdog, "verify_merge": verify_merge}


def launch_orchestration(
    *,
    job_name: str,
    shard_count: int,
    template_path: Path,
    output_root: Path,
    max_retries: int,
    dry_run: bool,
) -> Path:
    """Build shard artifacts and submit the first worker/watchdog wave.

    Examples:
        >>> callable(launch_orchestration)
        True
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    orchestration_root, shard_files = build_inference_shards(
        shard_count=shard_count,
        output_dir=str(output_root),
        job_name=job_name,
    )
    orchestration_root = Path(orchestration_root)
    base_config_path = Path(get_loaded_config_path())
    config_paths = _write_shard_configs(
        orchestration_root=orchestration_root,
        shard_files=shard_files,
        base_config_path=base_config_path,
    )
    slurm_scripts = _write_slurm_scripts(
        orchestration_root=orchestration_root,
        template_path=template_path,
        job_name=job_name,
    )
    manifest = _load_manifest(orchestration_root)
    manifest.update(
        {
            "base_config_path": str(base_config_path),
            "config_paths": [str(path) for path in config_paths],
            "run_dirs": [
                str(_expected_run_dir(orchestration_root, idx))
                for idx in range(len(shard_files))
            ],
            "slurm_scripts": {key: str(path) for key, path in slurm_scripts.items()},
        }
    )
    _write_json(orchestration_root / "manifest.json", manifest)
    status = _initial_status(
        orchestration_root=orchestration_root,
        shard_files=shard_files,
        max_retries=max_retries,
    )
    _write_json(orchestration_root / "status.json", status)
    worker_submission = _submit_sbatch(
        script_path=slurm_scripts["worker"],
        extra_args=["--array", _array_spec(list(range(len(shard_files))))],
        dry_run=dry_run,
    )
    status["worker_job_ids"].append(worker_submission["job_id"])
    watchdog_args = []
    if worker_submission["job_id"] is not None:
        watchdog_args = ["--dependency", f"afterany:{worker_submission['job_id']}"]
    watchdog_submission = _submit_sbatch(
        script_path=slurm_scripts["watchdog"],
        extra_args=watchdog_args,
        dry_run=dry_run,
    )
    status["watchdog_job_ids"].append(watchdog_submission["job_id"])
    _write_json(orchestration_root / "status.json", status)
    _write_submission(
        orchestration_root,
        {
            "worker_submission": worker_submission,
            "watchdog_submission": watchdog_submission,
            "dry_run": dry_run,
        },
    )
    return orchestration_root


def run_watchdog(*, orchestration_root: Path, dry_run: bool) -> None:
    """Retry incomplete shards or release final verification.

    Examples:
        >>> callable(run_watchdog)
        True
    """
    manifest = _load_manifest(orchestration_root)
    status = _refresh_status(orchestration_root)
    incomplete = [
        shard
        for shard in status["shards"]
        if shard["done_tiles"] < shard["expected_tiles"]
    ]
    if not incomplete:
        verify_submission = _submit_sbatch(
            script_path=Path(manifest["slurm_scripts"]["verify_merge"]),
            extra_args=None,
            dry_run=dry_run,
        )
        status["merge_job_id"] = verify_submission["job_id"]
        status["state"] = "verifying"
        _write_json(orchestration_root / "status.json", status)
        _write_json(
            orchestration_root / "submission.json",
            {
                **_load_json(orchestration_root / "submission.json"),
                "verify_merge_submission": verify_submission,
            },
        )
        return

    retryable = [
        shard
        for shard in incomplete
        if int(shard["retry_count"]) < int(status["max_retries"])
    ]
    if not retryable:
        status["state"] = "failed_incomplete"
        _write_json(orchestration_root / "status.json", status)
        raise SystemExit("incomplete shards remain and retry limit is exhausted")

    retry_indices = [int(shard["shard_id"]) for shard in retryable]
    for shard in status["shards"]:
        if int(shard["shard_id"]) in retry_indices:
            shard["retry_count"] = int(shard["retry_count"]) + 1
            shard["status"] = "retrying"
    worker_submission = _submit_sbatch(
        script_path=Path(manifest["slurm_scripts"]["worker"]),
        extra_args=["--array", _array_spec(retry_indices)],
        dry_run=dry_run,
    )
    status["worker_job_ids"].append(worker_submission["job_id"])
    watchdog_args = []
    if worker_submission["job_id"] is not None:
        watchdog_args = ["--dependency", f"afterany:{worker_submission['job_id']}"]
    watchdog_submission = _submit_sbatch(
        script_path=Path(manifest["slurm_scripts"]["watchdog"]),
        extra_args=watchdog_args,
        dry_run=dry_run,
    )
    status["watchdog_job_ids"].append(watchdog_submission["job_id"])
    status["state"] = "retry_submitted"
    _write_json(orchestration_root / "status.json", status)
    _write_json(
        orchestration_root / "submission.json",
        {
            **_load_json(orchestration_root / "submission.json"),
            "last_retry_worker_submission": worker_submission,
            "last_retry_watchdog_submission": watchdog_submission,
        },
    )


def run_verify_merge(*, orchestration_root: Path) -> None:
    """Verify shard completion and merge unions.

    Examples:
        >>> callable(run_verify_merge)
        True
    """
    status = _refresh_status(orchestration_root)
    incomplete = [
        shard
        for shard in status["shards"]
        if shard["done_tiles"] < shard["expected_tiles"]
    ]
    final_status_path = orchestration_root / "final_status.json"
    if incomplete:
        payload = {
            "status": "failed_incomplete",
            "incomplete_shards": [
                {
                    "shard_id": shard["shard_id"],
                    "done_tiles": shard["done_tiles"],
                    "expected_tiles": shard["expected_tiles"],
                }
                for shard in incomplete
            ],
        }
        _write_json(final_status_path, payload)
        raise SystemExit("verification failed: some shards are incomplete")

    merged_dir = orchestration_root / "merged"
    merged_paths = merge_shard_unions(
        shard_run_dirs=[str(Path(shard["run_dir"])) for shard in status["shards"]],
        output_dir=str(merged_dir),
    )
    payload = {
        "status": "success",
        "merged_paths": [str(path) for path in merged_paths],
        "merged_dir": str(merged_dir),
    }
    status["state"] = "complete"
    _write_json(orchestration_root / "status.json", status)
    _write_json(final_status_path, payload)


def main() -> None:
    """CLI entrypoint for shard orchestration.

    Examples:
        >>> callable(main)
        True
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", help="Name for the shard orchestration root.")
    parser.add_argument("--shards", type=int, help="Number of shard workers.")
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Read-only Slurm template file used to render worker/watchdog scripts.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where shard manifests, configs, scripts, and merged outputs live.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry submissions for incomplete shard runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render orchestration artifacts without calling sbatch.",
    )
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Internal mode: inspect shard progress and resubmit incomplete shards.",
    )
    parser.add_argument(
        "--verify-merge",
        action="store_true",
        help="Internal mode: verify all shards completed and merge unions.",
    )
    parser.add_argument(
        "--orchestration-root",
        help="Existing orchestration root for watchdog/verify-merge modes.",
    )
    args = parser.parse_args()

    if args.watchdog:
        if not args.orchestration_root:
            raise ValueError("--orchestration-root is required with --watchdog")
        run_watchdog(
            orchestration_root=Path(args.orchestration_root),
            dry_run=args.dry_run,
        )
        return

    if args.verify_merge:
        if not args.orchestration_root:
            raise ValueError("--orchestration-root is required with --verify-merge")
        run_verify_merge(orchestration_root=Path(args.orchestration_root))
        return

    if not args.job_name or not args.shards:
        raise ValueError("--job-name and --shards are required for launch mode")
    root = launch_orchestration(
        job_name=args.job_name,
        shard_count=args.shards,
        template_path=Path(args.template),
        output_root=Path(args.output_root),
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )
    print(f"orchestration root: {root}")


if __name__ == "__main__":
    main()
