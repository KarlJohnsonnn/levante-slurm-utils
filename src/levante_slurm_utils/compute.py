"""Machine-aware Dask and Slurm helpers for Levante-style workflows."""

from __future__ import annotations

import math
import os
import platform
import shutil
from dataclasses import dataclass
from typing import Iterable, Sequence

import dask
import xarray as xr

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    from dask.distributed import Client, LocalCluster, get_client
except Exception:  # pragma: no cover
    Client = None
    LocalCluster = None
    get_client = None

try:
    from dask_jobqueue import SLURMCluster
except Exception:  # pragma: no cover
    SLURMCluster = None


@dataclass(frozen=True)
class ScalingPlan:
    """Resource recommendation for a Slurm-backed Dask workload."""

    n_nodes: int
    n_cpu: int
    memory_gb: float
    n_workers: int
    walltime: str
    total_workload: int

    def as_tuple(self) -> tuple[int, int, float, int, str]:
        return self.n_nodes, self.n_cpu, self.memory_gb, self.n_workers, self.walltime


def is_server() -> bool:
    """Return True in server, JupyterHub, or Slurm-like environments."""
    if os.getenv("JUPYTERHUB_API_URL") or os.getenv("JUPYTERHUB_USER"):
        return True
    if os.getenv("SLURM_JOB_ID"):
        return True
    return platform.system() != "Darwin"


def in_slurm_allocation() -> bool:
    """Return True when running inside a Slurm allocation."""
    return bool(os.getenv("SLURM_JOB_ID"))


def _local_available_memory_bytes() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return int(8 * 1024**3)


def _dask_worker_memory_bytes() -> int | None:
    if get_client is None:
        return None
    try:
        info = get_client().scheduler_info()
    except Exception:
        return None
    workers = info.get("workers", {})
    limits = [int(w.get("memory_limit", 0)) for w in workers.values() if int(w.get("memory_limit", 0)) > 0]
    return min(limits) if limits else None


def recommend_target_chunk_mb(
    min_chunk_mb: int = 64,
    max_chunk_mb: int = 512,
    memory_fraction: float = 0.12,
) -> int:
    """Return chunk target in MB from active Dask worker or local memory."""
    if min_chunk_mb <= 0 or max_chunk_mb <= 0:
        raise ValueError("Chunk bounds must be positive.")
    if min_chunk_mb > max_chunk_mb:
        raise ValueError("min_chunk_mb must be <= max_chunk_mb.")
    if not (0.01 <= memory_fraction <= 0.8):
        raise ValueError("memory_fraction must be between 0.01 and 0.8.")

    memory_bytes = _dask_worker_memory_bytes() or _local_available_memory_bytes()
    target_mb = int((memory_bytes * memory_fraction) / (1024**2))
    return max(min_chunk_mb, min(max_chunk_mb, target_mb))


def _pick_reference_var(ds: xr.Dataset) -> xr.DataArray:
    if not ds.data_vars:
        raise ValueError("Dataset has no data variables.")
    return max(ds.data_vars.values(), key=lambda da: (da.ndim, da.size))


def _balanced_chunk_sizes(
    dim_sizes: dict[str, int],
    target_chunk_bytes: int,
    itemsize: int,
    prefer_dims: Iterable[str],
) -> dict[str, int]:
    dims = [dim for dim in prefer_dims if dim in dim_sizes] + [
        dim for dim in dim_sizes if dim not in set(prefer_dims)
    ]
    chunk = {dim: 1 for dim in dims}
    max_sizes = {dim: max(1, int(dim_sizes[dim])) for dim in dims}
    target_elems = max(1, int(target_chunk_bytes // max(1, itemsize)))
    current_elems = 1

    while True:
        grew = False
        for dim in dims:
            if chunk[dim] >= max_sizes[dim]:
                continue
            proposed = min(max_sizes[dim], chunk[dim] * 2)
            new_elems = (current_elems // chunk[dim]) * proposed
            if new_elems <= target_elems:
                chunk[dim] = proposed
                current_elems = new_elems
                grew = True
        if not grew:
            break

    if dims:
        first_dim = dims[0]
        if chunk[first_dim] < max_sizes[first_dim]:
            remaining = max(1, target_elems // max(1, current_elems // chunk[first_dim]))
            chunk[first_dim] = min(max_sizes[first_dim], max(chunk[first_dim], int(remaining)))
    return {dim: int(max(1, min(max_sizes[dim], value))) for dim, value in chunk.items()}


def auto_chunk_dataset(
    ds: xr.Dataset,
    *,
    target_chunk_mb: int | None = None,
    min_chunk_mb: int = 64,
    max_chunk_mb: int = 512,
    memory_fraction: float = 0.12,
    prefer_dims: tuple[str, ...] = ("time", "altitude", "latitude", "longitude", "diameter"),
) -> tuple[xr.Dataset, dict[str, int]]:
    """Return dataset rechunked to a memory-aware, balanced chunk plan."""
    ref = _pick_reference_var(ds)
    itemsize = max(1, int(ref.dtype.itemsize))
    target_mb = target_chunk_mb or recommend_target_chunk_mb(
        min_chunk_mb=min_chunk_mb,
        max_chunk_mb=max_chunk_mb,
        memory_fraction=memory_fraction,
    )
    chunk_dict = _balanced_chunk_sizes(
        dim_sizes=dict(ds.sizes),
        target_chunk_bytes=int(target_mb * 1024**2),
        itemsize=itemsize,
        prefer_dims=prefer_dims,
    )
    return ds.chunk(chunk_dict), chunk_dict


def describe_chunk_plan(ds: xr.Dataset, chunk_dict: dict[str, int]) -> str:
    """Return one-line description of chunk size, count, and dims."""
    ref = _pick_reference_var(ds)
    elems = 1
    for dim in ref.dims:
        elems *= chunk_dict.get(dim, ds.sizes[dim])
    chunk_mb = elems * max(1, int(ref.dtype.itemsize)) / (1024**2)
    n_chunks = 1
    for dim in ref.dims:
        n_chunks *= math.ceil(ds.sizes[dim] / chunk_dict.get(dim, ds.sizes[dim]))
    dims_txt = ", ".join(f"{dim}={chunk_dict.get(dim, ds.sizes[dim])}" for dim in ref.dims)
    return f"chunk ~{chunk_mb:.1f} MB, ~{n_chunks} chunks for '{ref.name}'; dims: {dims_txt}"


def scaling_plan(
    n_time_steps: int,
    n_experiments: int,
    n_stations: int,
    debug_mode: bool = False,
) -> ScalingPlan:
    """Return a Levante-shaped resource recommendation."""
    if min(n_time_steps, n_experiments, n_stations) < 0:
        raise ValueError("Workload dimensions must be non-negative.")
    if debug_mode:
        return ScalingPlan(1, 64, 32.0, 2, "00:10:00", n_time_steps * n_experiments * n_stations)

    total_workload = n_time_steps * n_experiments * n_stations
    base_cpu, base_memory, base_workers, base_walltime = 128, 64.0, 2, "02:00:00"
    if total_workload < 1e5:
        n_nodes, n_cpu, memory, workers, walltime = 1, base_cpu, base_memory, base_workers, base_walltime
    elif total_workload < 1e6:
        n_nodes, n_cpu, memory, workers, walltime = 2, base_cpu * 2, base_memory * 2, base_workers * 2, "06:00:00"
    elif total_workload < 1e7:
        n_nodes, n_cpu, memory, workers, walltime = 4, base_cpu * 2, base_memory * 3, base_workers * 4, "07:00:00"
    else:
        n_nodes, n_cpu, memory, workers, walltime = 8, base_cpu * 2, base_memory * 4, base_workers * 6, "08:00:00"

    if n_experiments > 50:
        workers = min(workers * 2, 32)
    if n_time_steps > 1000:
        memory = min(memory * 1.5, 512)
    return ScalingPlan(n_nodes, n_cpu, memory, workers, walltime, total_workload)


def calculate_optimal_scaling(
    n_time_steps: int,
    n_experiments: int,
    n_stations: int,
    debug_mode: bool = False,
) -> tuple[int, int, float, int, str]:
    """Return ``(n_nodes, n_cpu, memory_gb, n_workers, walltime)``."""
    plan = scaling_plan(n_time_steps, n_experiments, n_stations, debug_mode)
    print("Workload analysis:")
    print(f"  - Time steps: {n_time_steps}")
    print(f"  - Experiments: {n_experiments}")
    print(f"  - Stations: {n_stations}")
    print(f"  - Total workload estimate: {plan.total_workload}")
    print("Optimal scaling:")
    print(f"  - Nodes: {plan.n_nodes}")
    print(f"  - CPU per node: {plan.n_cpu}")
    print(f"  - Memory per node: {plan.memory_gb}GB")
    print(f"  - Scale up workers: {plan.n_workers}")
    print(f"  - Walltime: {plan.walltime}")
    return plan.as_tuple()


def _thread_exports(n_threads_per_process: int) -> list[str]:
    return [
        f"export OMP_NUM_THREADS={n_threads_per_process}",
        f"export MKL_NUM_THREADS={n_threads_per_process}",
        f"export OPENBLAS_NUM_THREADS={n_threads_per_process}",
        f"export VECLIB_MAXIMUM_THREADS={n_threads_per_process}",
        f"export NUMEXPR_NUM_THREADS={n_threads_per_process}",
    ]


def allocate_resources(
    n_cpu: int = 16,
    n_jobs: int = 1,
    m: int = 0,
    n_threads_per_process: int = 1,
    port: str | None = "7777",
    part: str = "compute",
    walltime: str = "02:00:00",
    account: str | None = None,
    python: str | None = None,
    name: str = "dask_cluster",
    log_dir: str = "./logs",
    job_script_prologue: Sequence[str] | None = None,
    conda_env: str | None = None,
    min_worker_memory_gb: float = 2.0,
) -> tuple:
    """Return ``(cluster, client)`` for a local or Slurm-backed Dask cluster."""
    if Client is None or LocalCluster is None:
        raise ImportError("Dask distributed unavailable. Install dask[distributed].")
    if n_cpu <= 0 or n_jobs <= 0:
        raise ValueError("n_cpu and n_jobs must be positive.")
    if n_threads_per_process <= 0:
        raise ValueError("n_threads_per_process must be >= 1.")
    if min_worker_memory_gb <= 0:
        raise ValueError("min_worker_memory_gb must be positive.")

    memory_per_node_gb = n_cpu if m == 0 else m
    requested_processes_per_node = max(1, n_cpu // n_threads_per_process)
    memory_capped_processes = max(1, int(memory_per_node_gb // min_worker_memory_gb))
    processes_per_node = min(requested_processes_per_node, memory_capped_processes)

    dask.config.set(
        {
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.terminate": 0.95,
            "array.slicing.split_large_chunks": True,
            "distributed.scheduler.worker-saturation": 0.95,
            "distributed.scheduler.worker-memory-limit": 0.95,
        }
    )

    if shutil.which("sbatch") is None:
        n_workers = min(n_cpu, os.cpu_count() or 4)
        memory_limit = f"{min(memory_per_node_gb, 512)}GB"
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
            dashboard_address=f":{port}" if port else None,
            name=name,
        )
        client = Client(cluster)
        print("sbatch not found; using local Dask cluster.")
        print(f"Workers: {n_workers}, memory limit: {memory_limit}")
        if port:
            print(f"Local dashboard: http://localhost:{port}")
        return cluster, client

    if SLURMCluster is None:
        raise ImportError("SLURMCluster unavailable. Install dask-jobqueue for HPC.")
    if not account:
        raise ValueError("account is required when sbatch is available.")

    prologue = list(job_script_prologue or [])
    if conda_env:
        prologue.extend(["source ~/.bashrc", f"conda activate {conda_env}"])
    prologue.extend(_thread_exports(n_threads_per_process))
    prologue.extend(["ulimit -s unlimited", "ulimit -c 0"])

    if processes_per_node < requested_processes_per_node:
        print(
            f"Memory-aware worker cap: {requested_processes_per_node} requested -> "
            f"{processes_per_node} processes/node "
            f"({memory_per_node_gb / processes_per_node:.2f} GB/worker)."
        )

    cluster = SLURMCluster(
        name=name,
        cores=n_cpu,
        processes=processes_per_node,
        n_workers=n_jobs,
        memory=f"{memory_per_node_gb}GB",
        account=account,
        queue=part,
        walltime=walltime,
        scheduler_options={"dashboard_address": f":{port}"} if port else {},
        job_extra_directives=[
            f"--output={log_dir}/%j.out",
            f"--error={log_dir}/%j.err",
            "--propagate=STACK",
        ],
        job_script_prologue=prologue,
        python=python,
    )
    if n_jobs > 1:
        cluster.scale(n_jobs)
    print(cluster.job_script())

    client = Client(cluster)
    if port:
        dashboard_address = cluster.scheduler_address
        host = dashboard_address.split("//")[-1].split(":")[0]
        print(f"Remote dashboard address: http://{host}:{port}")
        print(f"Setup ssh port forwarding: ssh -L {port}:{host}:{port} levante")
        print(f"Local dashboard address: http://localhost:{port}")
    return cluster, client
