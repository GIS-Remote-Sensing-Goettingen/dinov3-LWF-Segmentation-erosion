"""Workflow dispatch tests for the main pipeline entrypoint."""

from __future__ import annotations

import segedge.pipeline.run as run


def test_create_run_directories_copies_config_snapshot(tmp_path, monkeypatch):
    """New run directories should include a copy of the active config file.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(run.cfg.io.paths, "output_dir", str(tmp_path / "output"))
    monkeypatch.setattr(run.cfg.runtime, "resume_run", False)
    monkeypatch.setattr(run.cfg.runtime, "resume_run_dir", None)
    monkeypatch.setattr(run, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "configure_performance_logging",
        lambda *_args, **_kwargs: None,
    )

    run_paths = run._create_run_directories()

    snapshot_path = tmp_path / "output" / "run_001" / "config.yml"
    assert run_paths["run_dir"] == str(tmp_path / "output" / "run_001")
    assert snapshot_path.exists()


def test_main_dispatches_to_expected_workflow(monkeypatch):
    """`run.main()` should choose inference-only, manual, or LOO workflows.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(run, "time_start", lambda: 0.0)
    monkeypatch.setattr(run, "time_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_log_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "_create_run_directories",
        lambda: {
            "run_dir": "/tmp/run_001",
            "plot_dir": "/tmp/run_001/plots",
            "validation_plot_dir": "/tmp/run_001/plots/validation",
            "inference_plot_dir": "/tmp/run_001/plots/inference",
            "shape_dir": "/tmp/run_001/shapes",
        },
    )
    monkeypatch.setattr(run, "_load_processed_tiles", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        run,
        "_initialize_time_budget_state",
        lambda **_kwargs: {
            "enabled": False,
            "hours": 0.0,
            "scope": "training_only",
            "cutover_mode": "immediate_inference",
            "deadline_ts": None,
            "clock_start_ts": None,
            "cutover_triggered": False,
            "cutover_stage": "none",
        },
    )
    monkeypatch.setattr(run, "_log_run_configuration", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "_initialize_union_state",
        lambda **_kwargs: ({}, lambda *_args, **_kwargs_inner: None),
    )
    monkeypatch.setattr(
        run, "init_model", lambda *_args, **_kwargs: (None, None, "cpu")
    )
    monkeypatch.setattr(
        run, "_resolve_feature_cache", lambda *_args, **_kwargs: ("memory", None)
    )
    monkeypatch.setattr(
        run,
        "consolidate_cached_features",
        lambda **_kwargs: None,
    )

    monkeypatch.setattr(run.cfg.runtime, "resume_run", False)
    monkeypatch.setattr(run.cfg.model.backbone, "name", "test-backbone")
    monkeypatch.setattr(run.cfg.model.backbone, "patch_size", 14)
    monkeypatch.setattr(run.cfg.model.backbone, "resample_factor", 1.0)
    monkeypatch.setattr(run.cfg.model.tiling, "tile_size", 128)
    monkeypatch.setattr(run.cfg.model.tiling, "stride", 64)
    monkeypatch.setattr(run.cfg.model.banks, "feat_context_radius", 0)
    monkeypatch.setattr(run.cfg.io.inference, "model_bundle_dir", "/tmp/model_bundle")
    monkeypatch.setattr(run.cfg.io.inference, "save_bundle", False)
    monkeypatch.setattr(run.cfg.io.paths, "roads_mask_path", "/tmp/roads.tif")
    monkeypatch.setattr(run.cfg.io.paths, "source_label_raster", "/tmp/labels.tif")
    monkeypatch.setattr(run.cfg.io.paths, "eval_gt_vectors", [])
    monkeypatch.setattr(run.cfg.training.loo, "enabled", True)
    monkeypatch.setattr(run.cfg.training.loo, "min_train_tiles", 1)
    monkeypatch.setattr(run.cfg.training.loo, "val_tiles_per_fold", 1)
    monkeypatch.setattr(run.cfg.training.loo, "min_gt_positive_pixels", 0)
    monkeypatch.setattr(run.cfg.training.loo, "low_gt_policy", "skip_fold")

    branch_calls: list[str] = []

    def _record_call(name: str):
        """Return a workflow stub that records the selected branch.

        Examples:
            >>> callable(_record_call("demo"))
            True
        """

        def _runner(common, **_kwargs):
            """Record one workflow dispatch call.

            Examples:
                >>> True
                True
            """
            branch_calls.append(name)
            common["should_consolidate"] = False

        return _runner

    monkeypatch.setattr(run, "run_inference_only", _record_call("inference_only"))
    monkeypatch.setattr(run, "run_manual_training", _record_call("manual"))
    monkeypatch.setattr(run, "run_loo_training", _record_call("loo"))

    scenarios = [
        (
            "inference_only",
            False,
            False,
            {
                "auto_split_tiles": False,
                "gt_tiles": [],
                "source_tiles": [],
                "val_tiles": [],
                "holdout_tiles": ["holdout_a.tif"],
                "inference_dir": "/tmp/infer",
                "inference_glob": "*.tif",
            },
        ),
        (
            "manual",
            True,
            False,
            {
                "auto_split_tiles": False,
                "gt_tiles": [],
                "source_tiles": ["source_a.tif"],
                "val_tiles": ["val_a.tif"],
                "holdout_tiles": ["holdout_a.tif"],
                "inference_dir": "/tmp/infer",
                "inference_glob": "*.tif",
            },
        ),
        (
            "loo",
            True,
            True,
            {
                "auto_split_tiles": True,
                "gt_tiles": ["gt_a.tif", "gt_b.tif"],
                "source_tiles": [],
                "val_tiles": ["gt_a.tif", "gt_b.tif"],
                "holdout_tiles": ["holdout_a.tif"],
                "inference_dir": None,
                "inference_glob": "*.tif",
            },
        ),
    ]

    for expected, training_enabled, auto_split_enabled, tile_sets in scenarios:
        branch_calls.clear()
        monkeypatch.setattr(run.cfg.io, "training", training_enabled)
        monkeypatch.setattr(run.cfg.io.auto_split, "enabled", auto_split_enabled)
        monkeypatch.setattr(run, "_resolve_tile_sets", lambda **_kwargs: tile_sets)

        run.main()

        assert branch_calls == [expected]


def test_resolve_inference_model_bundle_dir_uses_latest_valid_previous_run(tmp_path):
    """Inference-only fallback should select the newest valid prior bundle.

    Examples:
        >>> True
        True
    """
    output_root = tmp_path / "output"
    current_run = output_root / "run_003"
    for run_name in ("run_001", "run_002", "run_003"):
        (output_root / run_name).mkdir(parents=True)
    (output_root / "run_001" / "model_bundle").mkdir()
    (output_root / "run_001" / "model_bundle" / "manifest.yml").write_text(
        "bundle_version: 1\n",
        encoding="utf-8",
    )
    (output_root / "run_002" / "model_bundle").mkdir()
    (output_root / "run_002" / "model_bundle" / "manifest.yml").write_text(
        "bundle_version: 1\n",
        encoding="utf-8",
    )

    resolved = run._resolve_inference_model_bundle_dir(
        None,
        output_root=str(output_root),
        current_run_dir=str(current_run),
    )

    assert resolved == str(output_root / "run_002" / "model_bundle")


def test_resolve_inference_model_bundle_dir_raises_without_previous_bundle(tmp_path):
    """Inference-only fallback should fail clearly when no bundle exists.

    Examples:
        >>> True
        True
    """
    output_root = tmp_path / "output"
    current_run = output_root / "run_001"
    current_run.mkdir(parents=True)

    try:
        run._resolve_inference_model_bundle_dir(
            None,
            output_root=str(output_root),
            current_run_dir=str(current_run),
        )
    except ValueError as exc:
        assert "no previous run bundle was found" in str(exc)
    else:
        raise AssertionError("expected ValueError when no previous bundle exists")


def test_main_resolves_previous_bundle_when_model_bundle_dir_is_null(monkeypatch):
    """Inference-only main path should pass the resolved fallback bundle.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(run, "time_start", lambda: 0.0)
    monkeypatch.setattr(run, "time_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "_log_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "_create_run_directories",
        lambda: {
            "run_dir": "/tmp/run_010",
            "plot_dir": "/tmp/run_010/plots",
            "validation_plot_dir": "/tmp/run_010/plots/validation",
            "inference_plot_dir": "/tmp/run_010/plots/inference",
            "shape_dir": "/tmp/run_010/shapes",
        },
    )
    monkeypatch.setattr(run, "_load_processed_tiles", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        run,
        "_initialize_time_budget_state",
        lambda **_kwargs: {
            "enabled": False,
            "hours": 0.0,
            "scope": "training_only",
            "cutover_mode": "immediate_inference",
            "deadline_ts": None,
            "clock_start_ts": None,
            "cutover_triggered": False,
            "cutover_stage": "none",
        },
    )
    monkeypatch.setattr(run, "_log_run_configuration", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "_initialize_union_state",
        lambda **_kwargs: ({}, lambda *_args, **_kwargs_inner: None),
    )
    monkeypatch.setattr(
        run, "init_model", lambda *_args, **_kwargs: (None, None, "cpu")
    )
    monkeypatch.setattr(
        run, "_resolve_feature_cache", lambda *_args, **_kwargs: ("memory", None)
    )
    monkeypatch.setattr(run, "consolidate_cached_features", lambda **_kwargs: None)
    monkeypatch.setattr(run.cfg.io, "training", False)
    monkeypatch.setattr(run.cfg.io.auto_split, "enabled", False)
    monkeypatch.setattr(run.cfg.io.inference, "model_bundle_dir", None)
    monkeypatch.setattr(run.cfg.io.inference, "save_bundle", False)
    monkeypatch.setattr(run.cfg.io.paths, "roads_mask_path", "/tmp/roads.tif")
    monkeypatch.setattr(run.cfg.io.paths, "source_label_raster", "/tmp/labels.tif")
    monkeypatch.setattr(run.cfg.io.paths, "eval_gt_vectors", [])
    monkeypatch.setattr(run.cfg.io.paths, "output_dir", "/tmp")
    monkeypatch.setattr(run.cfg.model.backbone, "name", "test-backbone")
    monkeypatch.setattr(run.cfg.model.backbone, "patch_size", 14)
    monkeypatch.setattr(run.cfg.model.backbone, "resample_factor", 1.0)
    monkeypatch.setattr(run.cfg.model.tiling, "tile_size", 128)
    monkeypatch.setattr(run.cfg.model.tiling, "stride", 64)
    monkeypatch.setattr(run.cfg.model.banks, "feat_context_radius", 0)
    monkeypatch.setattr(
        run,
        "_resolve_tile_sets",
        lambda **_kwargs: {
            "auto_split_tiles": False,
            "gt_tiles": [],
            "source_tiles": [],
            "val_tiles": [],
            "holdout_tiles": ["holdout_a.tif"],
            "inference_dir": "/tmp/infer",
            "inference_glob": "*.tif",
        },
    )
    monkeypatch.setattr(
        run,
        "_resolve_inference_model_bundle_dir",
        lambda *_args, **_kwargs: "/tmp/run_009/model_bundle",
    )

    seen_bundle_dirs: list[str] = []

    def _record_inference(common, **_kwargs):
        """Capture the resolved bundle path for inference-only dispatch.

        Examples:
            >>> True
            True
        """
        seen_bundle_dirs.append(common["model_bundle_dir"])
        common["should_consolidate"] = False

    monkeypatch.setattr(run, "run_inference_only", _record_inference)

    run.main()

    assert seen_bundle_dirs == ["/tmp/run_009/model_bundle"]
