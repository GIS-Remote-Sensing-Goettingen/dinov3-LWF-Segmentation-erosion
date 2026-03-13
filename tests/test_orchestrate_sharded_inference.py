"""Tests for Slurm shard orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import fiona
import yaml
from shapely.geometry import box, mapping

_ORCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "orchestrate_sharded_inference.py"
)
_ORCH_SPEC = importlib.util.spec_from_file_location(
    "orchestrate_sharded_inference", _ORCH_PATH
)
assert _ORCH_SPEC is not None
assert _ORCH_SPEC.loader is not None
_ORCH_MODULE = importlib.util.module_from_spec(_ORCH_SPEC)
sys.modules[_ORCH_SPEC.name] = _ORCH_MODULE
_ORCH_SPEC.loader.exec_module(_ORCH_MODULE)

launch_orchestration = _ORCH_MODULE.launch_orchestration
run_watchdog = _ORCH_MODULE.run_watchdog
run_verify_merge = _ORCH_MODULE.run_verify_merge


def _repo_config() -> dict:
    """Load the repository config fixture.

    Examples:
        >>> isinstance(_repo_config(), dict)
        True
    """
    cfg_path = Path(__file__).resolve().parents[1] / "config.yml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _write_template(path: Path) -> None:
    """Write a minimal Slurm template fixture.

    Examples:
        >>> True
        True
    """
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "#SBATCH --job-name=Erosion",
                "#SBATCH --output=segmentation_%j.out",
                "#SBATCH --error=segmentation_%j.err",
                "",
                "set -euo pipefail",
                "python -u ./main.py",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_fake_shards(root: Path, shard_count: int) -> tuple[Path, list[Path]]:
    """Create a deterministic shard manifest fixture.

    Examples:
        >>> True
        True
    """
    root.mkdir(parents=True, exist_ok=True)
    shard_files: list[Path] = []
    for idx in range(shard_count):
        shard_path = root / f"tiles_shard_{idx:03d}.txt"
        shard_path.write_text(
            f"tile_{idx}_a.tif\ntile_{idx}_b.tif\n",
            encoding="utf-8",
        )
        shard_files.append(shard_path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "assignment_method": "round_robin",
                "job_name": root.name,
                "num_shards": shard_count,
                "tiles_file_paths": [str(path) for path in shard_files],
                "total_tiles": shard_count * 2,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, shard_files


def _write_union(path: Path, fid: int) -> None:
    """Write a one-feature union shapefile fixture.

    Examples:
        >>> True
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(
        path,
        "w",
        driver="ESRI Shapefile",
        crs={"init": "epsg:4326"},
        schema=schema,
    ) as dst:
        dst.write(
            {
                "geometry": mapping(box(float(fid), 0.0, float(fid + 1), 1.0)),
                "properties": {"id": fid},
            }
        )


def test_launch_orchestration_dry_run_writes_configs_and_scripts(tmp_path, monkeypatch):
    """Dry-run launch should render configs, scripts, and submission metadata.

    Examples:
        >>> True
        True
    """
    template_path = tmp_path / "template.sh"
    _write_template(template_path)
    base_cfg_path = tmp_path / "base.yml"
    base_cfg_path.write_text(
        yaml.safe_dump(_repo_config(), sort_keys=False),
        encoding="utf-8",
    )

    def fake_build(*, shard_count: int, output_dir: str, job_name: str):
        return _write_fake_shards(Path(output_dir) / job_name, shard_count)

    monkeypatch.setattr(_ORCH_MODULE, "build_inference_shards", fake_build)
    monkeypatch.setattr(
        _ORCH_MODULE, "get_loaded_config_path", lambda: str(base_cfg_path)
    )

    root = launch_orchestration(
        job_name="demo",
        shard_count=2,
        template_path=template_path,
        output_root=tmp_path / "shards",
        max_retries=3,
        dry_run=True,
    )

    shard_cfg = yaml.safe_load((root / "configs" / "shard_000.yml").read_text())
    assert shard_cfg["io"]["inference"]["tiles_file"].endswith("tiles_shard_000.txt")
    assert shard_cfg["runtime"]["resume_run"] is True
    assert shard_cfg["runtime"]["run_dir"].endswith("runs/shard_000")
    worker_script = (root / "slurm" / "worker_array.sh").read_text(encoding="utf-8")
    assert "--config" in worker_script
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    assert submission["dry_run"] is True
    assert submission["worker_submission"]["command"][0] == "sbatch"


def test_watchdog_resubmits_only_incomplete_shards(tmp_path, monkeypatch):
    """Watchdog should resubmit only shard indices that are still incomplete.

    Examples:
        >>> True
        True
    """
    root = tmp_path / "orchestration"
    _write_fake_shards(root, 2)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for name in ("worker", "watchdog", "verify_merge"):
        (slurm_dir / f"{name}.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    manifest["slurm_scripts"] = {
        "worker": str(slurm_dir / "worker.sh"),
        "watchdog": str(slurm_dir / "watchdog.sh"),
        "verify_merge": str(slurm_dir / "verify_merge.sh"),
    }
    _ORCH_MODULE._write_json(root / "manifest.json", manifest)
    _ORCH_MODULE._write_json(
        root / "status.json",
        {
            "max_retries": 2,
            "shards": [
                {
                    "shard_id": 0,
                    "tiles_file": str(root / "tiles_shard_000.txt"),
                    "run_dir": str(root / "runs" / "shard_000"),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                },
                {
                    "shard_id": 1,
                    "tiles_file": str(root / "tiles_shard_001.txt"),
                    "run_dir": str(root / "runs" / "shard_001"),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                },
            ],
            "worker_job_ids": [],
            "watchdog_job_ids": [],
            "merge_job_id": None,
            "state": "submitted",
        },
    )
    (root / "runs" / "shard_000").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "shard_001").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "shard_000" / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_0_a.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_0_b.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "runs" / "shard_001" / "processed_tiles.jsonl").write_text(
        json.dumps({"tile_path": "tile_1_a.tif", "status": "done"}) + "\n",
        encoding="utf-8",
    )

    calls: list[dict] = []

    def fake_submit(*, script_path: Path, extra_args, dry_run: bool):
        calls.append(
            {
                "script_path": str(script_path),
                "extra_args": extra_args,
                "dry_run": dry_run,
            }
        )
        return {
            "job_id": str(100 + len(calls)),
            "command": ["sbatch", str(script_path)],
        }

    monkeypatch.setattr(_ORCH_MODULE, "_submit_sbatch", fake_submit)

    run_watchdog(orchestration_root=root, dry_run=False)

    assert calls[0]["extra_args"] == ["--array", "1"]
    assert calls[1]["extra_args"] == ["--dependency", "afterany:101"]
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    shard_one = next(shard for shard in status["shards"] if shard["shard_id"] == 1)
    assert shard_one["retry_count"] == 1


def test_verify_merge_requires_completion_and_writes_final_status(tmp_path):
    """Final verification should merge only after all shards complete.

    Examples:
        >>> True
        True
    """
    root = tmp_path / "orchestration"
    _write_fake_shards(root, 2)
    run_a = root / "runs" / "shard_000"
    run_b = root / "runs" / "shard_001"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (run_a / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_0_a.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_0_b.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_b / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_1_a.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_1_b.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for variant in ("raw", "shadow_with_proposals"):
        _write_union(run_a / "shapes" / "unions" / variant / "union.shp", 1)
        _write_union(run_b / "shapes" / "unions" / variant / "union.shp", 2)
    _ORCH_MODULE._write_json(
        root / "status.json",
        {
            "max_retries": 1,
            "shards": [
                {
                    "shard_id": 0,
                    "tiles_file": str(root / "tiles_shard_000.txt"),
                    "run_dir": str(run_a),
                    "expected_tiles": 2,
                    "done_tiles": 2,
                    "retry_count": 0,
                    "status": "complete",
                },
                {
                    "shard_id": 1,
                    "tiles_file": str(root / "tiles_shard_001.txt"),
                    "run_dir": str(run_b),
                    "expected_tiles": 2,
                    "done_tiles": 2,
                    "retry_count": 0,
                    "status": "complete",
                },
            ],
            "worker_job_ids": [],
            "watchdog_job_ids": [],
            "merge_job_id": None,
            "state": "verifying",
        },
    )

    run_verify_merge(orchestration_root=root)

    final_status = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
    assert final_status["status"] == "success"
    assert (root / "merged" / "raw" / "union.shp").exists()


def test_watchdog_submits_verify_when_all_shards_are_complete(tmp_path, monkeypatch):
    """Watchdog should release final verification immediately when all shards are done.

    Examples:
        >>> True
        True
    """
    root = tmp_path / "orchestration"
    _write_fake_shards(root, 1)
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for name in ("worker", "watchdog", "verify_merge"):
        (slurm_dir / f"{name}.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _ORCH_MODULE._write_json(
        root / "manifest.json",
        {
            "slurm_scripts": {
                "worker": str(slurm_dir / "worker.sh"),
                "watchdog": str(slurm_dir / "watchdog.sh"),
                "verify_merge": str(slurm_dir / "verify_merge.sh"),
            }
        },
    )
    run_dir = root / "runs" / "shard_000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_0_a.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_0_b.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _ORCH_MODULE._write_json(
        root / "status.json",
        {
            "max_retries": 2,
            "shards": [
                {
                    "shard_id": 0,
                    "tiles_file": str(root / "tiles_shard_000.txt"),
                    "run_dir": str(run_dir),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                }
            ],
            "worker_job_ids": [],
            "watchdog_job_ids": [],
            "merge_job_id": None,
            "state": "submitted",
        },
    )

    calls: list[dict] = []

    def fake_submit(*, script_path: Path, extra_args, dry_run: bool):
        calls.append({"script_path": str(script_path), "extra_args": extra_args})
        return {"job_id": "501", "command": ["sbatch", str(script_path)]}

    monkeypatch.setattr(_ORCH_MODULE, "_submit_sbatch", fake_submit)

    run_watchdog(orchestration_root=root, dry_run=False)

    assert len(calls) == 1
    assert calls[0]["script_path"].endswith("verify_merge.sh")
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "verifying"
    assert status["merge_job_id"] == "501"


def test_watchdog_fails_when_retry_budget_is_exhausted(tmp_path):
    """Watchdog should stop once incomplete shards exceed the retry budget.

    Examples:
        >>> True
        True
    """
    root = tmp_path / "orchestration"
    _write_fake_shards(root, 1)
    _ORCH_MODULE._write_json(
        root / "manifest.json",
        {"slurm_scripts": {"worker": "", "watchdog": "", "verify_merge": ""}},
    )
    _ORCH_MODULE._write_json(
        root / "status.json",
        {
            "max_retries": 0,
            "shards": [
                {
                    "shard_id": 0,
                    "tiles_file": str(root / "tiles_shard_000.txt"),
                    "run_dir": str(root / "runs" / "shard_000"),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                }
            ],
            "worker_job_ids": [],
            "watchdog_job_ids": [],
            "merge_job_id": None,
            "state": "submitted",
        },
    )

    try:
        run_watchdog(orchestration_root=root, dry_run=False)
    except SystemExit as exc:
        assert "retry limit" in str(exc)
    else:
        raise AssertionError("expected exhausted retry budget to stop the watchdog")

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed_incomplete"


def test_verify_merge_fails_and_writes_status_when_incomplete(tmp_path):
    """Final verification should fail loudly when any shard is incomplete.

    Examples:
        >>> True
        True
    """
    root = tmp_path / "orchestration"
    _write_fake_shards(root, 1)
    _ORCH_MODULE._write_json(
        root / "status.json",
        {
            "max_retries": 1,
            "shards": [
                {
                    "shard_id": 0,
                    "tiles_file": str(root / "tiles_shard_000.txt"),
                    "run_dir": str(root / "runs" / "shard_000"),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "incomplete",
                }
            ],
            "worker_job_ids": [],
            "watchdog_job_ids": [],
            "merge_job_id": None,
            "state": "verifying",
        },
    )

    try:
        run_verify_merge(orchestration_root=root)
    except SystemExit as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected verify_merge to fail on incomplete shards")

    final_status = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
    assert final_status["status"] == "failed_incomplete"
