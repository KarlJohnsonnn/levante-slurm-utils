from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from levante_slurm_utils import (  # noqa: E402
    auto_chunk_dataset,
    calculate_optimal_scaling,
    describe_chunk_plan,
    recommend_target_chunk_mb,
)
from levante_slurm_utils.cli import main as cli_main  # noqa: E402
from levante_slurm_utils.compute import allocate_resources, scaling_plan  # noqa: E402


def test_scaling_tiers_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert calculate_optimal_scaling(10, 10, 10) == (1, 128, 64.0, 2, "02:00:00")

    plan = scaling_plan(1200, 80, 200)
    assert plan.n_nodes == 8
    assert plan.memory_gb == 384.0
    assert plan.n_workers == 24

    assert cli_main(["--time-steps", "10", "--experiments", "2", "--stations", "3"]) == 0
    assert "n_nodes=1" in capsys.readouterr().out


def test_chunk_helpers() -> None:
    ds = xr.Dataset(
        {"q": (("time", "altitude", "latitude"), np.ones((10, 4, 3), dtype="f4"))}
    )
    target = recommend_target_chunk_mb(min_chunk_mb=1, max_chunk_mb=2, memory_fraction=0.01)
    chunked, chunks = auto_chunk_dataset(ds, target_chunk_mb=1)
    text = describe_chunk_plan(chunked, chunks)

    assert 1 <= target <= 2
    assert set(chunks) == {"time", "altitude", "latitude"}
    assert "chunk ~" in text


def test_allocate_local_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("distributed")
    original_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "sbatch" else original_which(name))

    cluster, client = allocate_resources(n_cpu=1, n_jobs=1, m=1, port=None)
    try:
        assert len(client.scheduler_info()["workers"]) == 1
    finally:
        client.close()
        cluster.close()
