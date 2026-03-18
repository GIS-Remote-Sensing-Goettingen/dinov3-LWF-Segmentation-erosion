"""Tests for simple batch-per-job Slurm inference orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

_BATCH_PATH = (
    Path(__file__).resolve().parents[1] / "deployment" / "launch_batched_inference.py"
)
_BATCH_SPEC = importlib.util.spec_from_file_location(
    "launch_batched_inference", _BATCH_PATH
)
assert _BATCH_SPEC is not None
assert _BATCH_SPEC.loader is not None
_BATCH_MODULE = importlib.util.module_from_spec(_BATCH_SPEC)
sys.modules[_BATCH_SPEC.name] = _BATCH_MODULE
_BATCH_SPEC.loader.exec_module(_BATCH_MODULE)

launch_batched_inference = _BATCH_MODULE.launch_batched_inference
run_controller = _BATCH_MODULE.run_controller


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
                'cd "${SLURM_SUBMIT_DIR:-$PWD}"',
                " cd /user/davide.mattioli/u20330/dinov3-LWF-Segmentation-erosion",
                "python -u ./main.py",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_fake_batches(root: Path, batch_sizes: list[int]) -> tuple[Path, list[Path]]:
    """Create a deterministic batch manifest fixture.

    Examples:
        >>> True
        True
    """
    root.mkdir(parents=True, exist_ok=True)
    batch_files: list[Path] = []
    total_tiles = 0
    for idx, size in enumerate(batch_sizes):
        batch_path = root / f"tiles_batch_{idx:03d}.txt"
        batch_path.write_text(
            "".join(f"tile_{idx}_{tile_idx}.tif\n" for tile_idx in range(size)),
            encoding="utf-8",
        )
        batch_files.append(batch_path)
        total_tiles += size
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "assignment_method": "contiguous_chunks",
                "batch_size": max(batch_sizes) if batch_sizes else 0,
                "job_name": root.name,
                "num_batches": len(batch_sizes),
                "tiles_file_paths": [str(path) for path in batch_files],
                "tiles_per_batch": batch_sizes,
                "total_tiles": total_tiles,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, batch_files


def _write_union(path: Path, fid: int, *, x_origin: float = 0.0) -> None:
    """Write a one-stage union raster fixture.

    Examples:
        >>> True
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.array([[1, 0] if fid == 1 else [0, 1]], dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pixels.shape[1],
        height=pixels.shape[0],
        count=1,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=from_origin(float(x_origin), 1.0, 1.0, 1.0),
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(pixels, 1)


def test_launch_batched_inference_dry_run_writes_configs_and_scripts(
    tmp_path,
    monkeypatch,
):
    """Dry-run launch should render per-batch configs, scripts, and submissions.

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

    def fake_build(*, batch_size: int, output_dir: Path, job_name: str):
        assert batch_size == 100
        return _write_fake_batches(output_dir / job_name, [100, 25])

    monkeypatch.setattr(_BATCH_MODULE, "build_inference_batches", fake_build)
    monkeypatch.setattr(
        _BATCH_MODULE, "get_loaded_config_path", lambda: str(base_cfg_path)
    )

    root = launch_batched_inference(
        job_name="demo",
        batch_size=100,
        template_path=template_path,
        output_root=tmp_path / "batches",
        max_retries=3,
        dry_run=True,
    )

    batch_cfg = yaml.safe_load((root / "configs" / "batch_000.yml").read_text())
    assert batch_cfg["io"]["inference"]["tiles_file"].endswith("tiles_batch_000.txt")
    assert batch_cfg["io"]["inference"]["tiles_dir"] is None
    assert batch_cfg["runtime"]["resume_run"] is True
    assert batch_cfg["runtime"]["run_dir"].endswith("runs/batch_000")
    worker_script = (root / "slurm" / "worker_batch_000.sh").read_text(encoding="utf-8")
    assert f"REPO_ROOT={_BATCH_MODULE.REPO_ROOT}" in worker_script
    assert 'cd "${REPO_ROOT}"' in worker_script
    assert "--config" in worker_script
    assert "SLURM_ARRAY_TASK_ID" not in worker_script
    assert (
        "cd /user/davide.mattioli/u20330/dinov3-LWF-Segmentation-erosion"
        not in worker_script
    )
    controller_script = (root / "slurm" / "controller.sh").read_text(encoding="utf-8")
    assert "--controller" in controller_script
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    assert submission["dry_run"] is True
    assert len(submission["worker_submissions"]) == 2
    assert submission["worker_submissions"][0]["submission"]["command"][0] == "sbatch"


def test_controller_resubmits_only_incomplete_batches(tmp_path, monkeypatch):
    """Controller should resubmit only the batches that are still incomplete.

    Examples:
        >>> True
        True
    """
    root, batch_files = _write_fake_batches(tmp_path / "orchestration", [2, 2])
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    worker_a = slurm_dir / "worker_batch_000.sh"
    worker_b = slurm_dir / "worker_batch_001.sh"
    worker_a.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    worker_b.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    controller = slurm_dir / "controller.sh"
    controller.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _BATCH_MODULE._write_json(
        root / "manifest.json",
        {
            "controller_script": str(controller),
            "slurm_scripts": {
                "controller": str(controller),
                "workers": [str(worker_a), str(worker_b)],
            },
        },
    )
    run_a = root / "runs" / "batch_000"
    run_b = root / "runs" / "batch_001"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (run_a / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_0_0.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_0_1.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_b / "processed_tiles.jsonl").write_text(
        json.dumps({"tile_path": "tile_1_0.tif", "status": "done"}) + "\n",
        encoding="utf-8",
    )
    _BATCH_MODULE._write_json(
        root / "status.json",
        {
            "batch_size": 100,
            "batches": [
                {
                    "batch_id": 0,
                    "tiles_file": str(batch_files[0]),
                    "worker_script": str(worker_a),
                    "run_dir": str(run_a),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                },
                {
                    "batch_id": 1,
                    "tiles_file": str(batch_files[1]),
                    "worker_script": str(worker_b),
                    "run_dir": str(run_b),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                },
            ],
            "controller_job_ids": [],
            "max_retries": 2,
            "state": "submitted",
            "worker_job_ids": [],
        },
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

    monkeypatch.setattr(_BATCH_MODULE, "_submit_sbatch", fake_submit)

    run_controller(orchestration_root=root, dry_run=False)

    assert calls[0]["script_path"].endswith("worker_batch_001.sh")
    assert calls[0]["extra_args"] is None
    assert calls[1]["script_path"].endswith("controller.sh")
    assert calls[1]["extra_args"] == ["--dependency", "afterany:101"]
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    batch_one = next(batch for batch in status["batches"] if batch["batch_id"] == 1)
    assert batch_one["retry_count"] == 1
    assert batch_one["last_worker_job_id"] == "101"
    assert batch_one["worker_stderr_path"].endswith("worker_batch_001_101.err")
    assert batch_one["worker_stdout_path"].endswith("worker_batch_001_101.out")


def test_controller_merges_when_all_batches_are_complete(tmp_path):
    """Controller should merge unions immediately when all batches are complete.

    Examples:
        >>> True
        True
    """
    root, batch_files = _write_fake_batches(tmp_path / "orchestration", [2, 2])
    run_a = root / "runs" / "batch_000"
    run_b = root / "runs" / "batch_001"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (run_a / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_0_0.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_0_1.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_b / "processed_tiles.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tile_path": "tile_1_0.tif", "status": "done"}),
                json.dumps({"tile_path": "tile_1_1.tif", "status": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for variant in ("raw", "shadow_with_proposals"):
        _write_union(run_a / "shapes" / "unions" / variant / "union.tif", 1)
        _write_union(
            run_b / "shapes" / "unions" / variant / "union.tif",
            2,
            x_origin=2.0,
        )
    _BATCH_MODULE._write_json(
        root / "status.json",
        {
            "batch_size": 100,
            "batches": [
                {
                    "batch_id": 0,
                    "tiles_file": str(batch_files[0]),
                    "worker_script": str(root / "slurm" / "worker_batch_000.sh"),
                    "run_dir": str(run_a),
                    "expected_tiles": 2,
                    "done_tiles": 2,
                    "retry_count": 0,
                    "status": "complete",
                },
                {
                    "batch_id": 1,
                    "tiles_file": str(batch_files[1]),
                    "worker_script": str(root / "slurm" / "worker_batch_001.sh"),
                    "run_dir": str(run_b),
                    "expected_tiles": 2,
                    "done_tiles": 2,
                    "retry_count": 0,
                    "status": "complete",
                },
            ],
            "controller_job_ids": [],
            "max_retries": 1,
            "state": "submitted",
            "worker_job_ids": [],
        },
    )

    run_controller(orchestration_root=root, dry_run=False)

    final_status = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
    assert final_status["status"] == "success"
    assert (root / "merged" / "raw" / "union.tif").exists()
    with rasterio.open(root / "merged" / "raw" / "union.tif") as src:
        merged_pixels = src.read(1)
    np.testing.assert_array_equal(
        merged_pixels,
        np.array([[1, 0, 0, 1]], dtype=np.uint8),
    )


def test_submit_controller_only_submits_existing_controller_script(
    tmp_path,
    monkeypatch,
):
    """Submit-controller mode should submit only the existing controller script.

    Examples:
        >>> True
        True
    """
    root, _batch_files = _write_fake_batches(tmp_path / "orchestration", [2])
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    controller = slurm_dir / "controller.sh"
    controller.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _BATCH_MODULE._write_json(
        root / "manifest.json",
        {"controller_script": str(controller)},
    )
    _BATCH_MODULE._write_json(
        root / "status.json",
        {
            "batch_size": 100,
            "batches": [],
            "controller_job_ids": [],
            "max_retries": 1,
            "state": "submitted",
            "worker_job_ids": [],
        },
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
            "job_id": "901",
            "command": ["sbatch", str(script_path)],
        }

    monkeypatch.setattr(_BATCH_MODULE, "_submit_sbatch", fake_submit)

    submission = _BATCH_MODULE.submit_controller_only(
        orchestration_root=root,
        dry_run=False,
    )

    assert submission["job_id"] == "901"
    assert calls == [
        {
            "script_path": str(controller),
            "extra_args": None,
            "dry_run": False,
        }
    ]
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["controller_job_ids"] == ["901"]
    assert status["state"] == "controller_submitted"
    submission_json = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    assert submission_json["manual_controller_submission"]["job_id"] == "901"


def test_main_requires_orchestration_root_with_submit_controller(monkeypatch):
    """CLI should require an orchestration root for submit-controller mode.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_batched_inference.py", "--submit-controller"],
    )

    try:
        _BATCH_MODULE.main()
    except ValueError as exc:
        assert "--orchestration-root" in str(exc)
    else:
        raise AssertionError("expected missing orchestration root to fail")


def test_controller_fails_when_retry_budget_is_exhausted(tmp_path):
    """Controller should stop once incomplete batches exceed the retry budget.

    Examples:
        >>> True
        True
    """
    root, batch_files = _write_fake_batches(tmp_path / "orchestration", [2])
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    worker_err = slurm_dir / "worker_batch_000_700.err"
    worker_err.write_text(
        (
            "bash: line 27: cd: "
            "/user/davide.mattioli/u20330/dinov3-LWF-Segmentation-erosion: "
            "No such file or directory\n"
        ),
        encoding="utf-8",
    )
    _BATCH_MODULE._write_json(
        root / "manifest.json",
        {"controller_script": str(slurm_dir / "controller.sh")},
    )
    _BATCH_MODULE._write_json(
        root / "status.json",
        {
            "batch_size": 100,
            "batches": [
                {
                    "batch_id": 0,
                    "tiles_file": str(batch_files[0]),
                    "worker_script": str(slurm_dir / "worker_batch_000.sh"),
                    "run_dir": str(root / "runs" / "batch_000"),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                    "last_worker_job_id": "700",
                    "worker_stdout_path": str(slurm_dir / "worker_batch_000_700.out"),
                    "worker_stderr_path": str(worker_err),
                }
            ],
            "controller_job_ids": [],
            "max_retries": 0,
            "state": "submitted",
            "worker_job_ids": [],
        },
    )

    try:
        run_controller(orchestration_root=root, dry_run=False)
    except SystemExit as exc:
        assert "retry limit" in str(exc)
    else:
        raise AssertionError("expected exhausted retry budget to stop the controller")

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed_incomplete"
    batch = status["batches"][0]
    assert batch["last_failure_source"] == "worker_stderr"
    assert "No such file or directory" in batch["last_failure_reason"]
    final_status = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
    assert final_status["status"] == "failed_incomplete"
    assert (
        final_status["incomplete_batches"][0]["last_failure_source"] == "worker_stderr"
    )


def test_controller_marks_missing_run_dir_when_logs_are_absent(tmp_path):
    """Controller should record a startup hint when a batch never created its run dir.

    Examples:
        >>> True
        True
    """
    root, batch_files = _write_fake_batches(tmp_path / "orchestration", [2])
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    _BATCH_MODULE._write_json(
        root / "manifest.json",
        {"controller_script": str(slurm_dir / "controller.sh")},
    )
    _BATCH_MODULE._write_json(
        root / "status.json",
        {
            "batch_size": 100,
            "batches": [
                {
                    "batch_id": 0,
                    "tiles_file": str(batch_files[0]),
                    "worker_script": str(slurm_dir / "worker_batch_000.sh"),
                    "run_dir": str(root / "runs" / "batch_000"),
                    "expected_tiles": 2,
                    "done_tiles": 0,
                    "retry_count": 0,
                    "status": "pending",
                }
            ],
            "controller_job_ids": [],
            "max_retries": 0,
            "state": "submitted",
            "worker_job_ids": [],
        },
    )

    try:
        run_controller(orchestration_root=root, dry_run=False)
    except SystemExit as exc:
        assert "retry limit" in str(exc)
    else:
        raise AssertionError("expected exhausted retry budget to stop the controller")

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    batch = status["batches"][0]
    assert batch["run_dir_exists"] is False
    assert batch["worker_logs_present"] is False
    assert batch["last_failure_source"] == "status"
    assert batch["last_failure_reason"] == "run_dir_missing"
