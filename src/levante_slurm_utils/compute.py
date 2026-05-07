"""Machine-aware Dask and Slurm helpers for Levante-style workflows."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import dask.config as dask_config
import xarray as xr

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    from dask.distributed import Client, LocalCluster, get_client, get_task_stream, performance_report
    from distributed.diagnostics.memory_sampler import MemorySampler
except Exception:  # pragma: no cover
    Client = None
    LocalCluster = None
    get_client = None
    get_task_stream = None
    performance_report = None
    MemorySampler = None

try:
    from dask_jobqueue.slurm import SLURMCluster
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


def _get_active_client(client: Any | None = None) -> Any | None:
    if client is not None:
        return client
    if get_client is None:
        return None
    try:
        return get_client()
    except Exception:
        return None


def dask_cluster_snapshot(client: Any | None = None) -> dict[str, Any]:
    """Return scheduler and worker state useful when the dashboard looks idle."""
    active = _get_active_client(client)
    if active is None:
        return {"client": None, "workers": 0}
    try:
        info = active.scheduler_info()
    except Exception as exc:
        return {"client": repr(active), "error": repr(exc)}

    workers = info.get("workers", {})
    worker_rows = []
    for address, worker in workers.items():
        metrics = worker.get("metrics", {})
        worker_rows.append(
            {
                "address": address,
                "name": worker.get("name"),
                "nthreads": worker.get("nthreads"),
                "memory_limit": worker.get("memory_limit"),
                "memory": metrics.get("memory"),
                "managed_bytes": metrics.get("managed_bytes"),
                "spilled_bytes": metrics.get("spilled_bytes"),
                "executing": metrics.get("executing"),
            }
        )
    return {
        "scheduler": info.get("address"),
        "dashboard": info.get("services", {}).get("dashboard"),
        "workers": len(workers),
        "threads": sum(int(w.get("nthreads", 0)) for w in workers.values()),
        "worker_rows": worker_rows,
    }


@dataclass
class DaskProfileArtifacts:
    """Paths written by ``DaskProfiler``."""

    output_dir: Path
    phase_timings_json: Path
    scheduler_before_json: Path
    scheduler_after_json: Path
    scheduler_ready_json: Path | None = None
    task_stream_json: Path | None = None
    memory_csv: Path | None = None
    performance_report_html: Path | None = None
    worker_profile_json: Path | None = None
    scheduler_profile_json: Path | None = None


class DaskProfiler:
    """Opt-in profiler for notebook phases and distributed computations."""

    def __init__(
        self,
        label: str,
        output_dir: str | os.PathLike[str] = "logs/dask-profile",
        *,
        client: Any | None = None,
        enabled: bool = True,
        performance: bool = True,
        memory: bool = True,
        task_stream: bool = True,
        statistical_profile: bool = True,
        memory_interval: float = 0.5,
    ) -> None:
        self.label = label
        self.output_dir = Path(output_dir)
        self.client = _get_active_client(client)
        self.enabled = bool(enabled)
        self.performance = bool(performance)
        self.memory = bool(memory)
        self.task_stream = bool(task_stream)
        self.statistical_profile = bool(statistical_profile)
        self.memory_interval = float(memory_interval)
        self.timings: list[dict[str, Any]] = []
        self.artifacts: DaskProfileArtifacts | None = None
        self._stack: ExitStack | None = None
        self._task_stream_cm: Any | None = None
        self._memory_sampler: Any | None = None
        self._collectors_started = False
        self._profile_start: float | None = None

    def __enter__(self) -> DaskProfiler:
        if not self.enabled:
            return self
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"{stamp}_{self.label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = DaskProfileArtifacts(
            output_dir=run_dir,
            phase_timings_json=run_dir / "phase_timings.json",
            scheduler_before_json=run_dir / "scheduler_before.json",
            scheduler_after_json=run_dir / "scheduler_after.json",
            scheduler_ready_json=run_dir / "scheduler_ready.json",
            task_stream_json=run_dir / "task_stream.json" if self.task_stream else None,
            memory_csv=run_dir / "memory.csv" if self.memory else None,
            performance_report_html=run_dir / "dask-report.html" if self.performance else None,
            worker_profile_json=run_dir / "profile_workers.json" if self.statistical_profile else None,
            scheduler_profile_json=run_dir / "profile_scheduler.json" if self.statistical_profile else None,
        )
        _write_json(self.artifacts.scheduler_before_json, dask_cluster_snapshot(self.client))
        self._stack = ExitStack()
        self._ensure_collectors()
        return self

    def _ensure_collectors(self) -> None:
        if self._collectors_started or not self.enabled or self.artifacts is None or self._stack is None:
            return
        self.client = _get_active_client(self.client)
        if self.client is None:
            return
        if self.artifacts.scheduler_ready_json is not None:
            _write_json(self.artifacts.scheduler_ready_json, dask_cluster_snapshot(self.client))
        if self.performance and performance_report is not None and self.artifacts.performance_report_html is not None:
            self._stack.enter_context(performance_report(filename=str(self.artifacts.performance_report_html)))
        if self.memory and MemorySampler is not None:
            self._memory_sampler = MemorySampler()
            self._stack.enter_context(
                self._memory_sampler.sample(self.label, client=self.client, interval=self.memory_interval)
            )
        if self.task_stream and get_task_stream is not None:
            self._task_stream_cm = self._stack.enter_context(get_task_stream(client=self.client))
        self._profile_start = time.time()
        self._collectors_started = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.enabled or self.artifacts is None:
            return None
        profile_stop = time.time()
        if self._stack is not None:
            self._stack.__exit__(exc_type, exc, tb)

        if self._task_stream_cm is not None and self.artifacts.task_stream_json is not None:
            _write_json(self.artifacts.task_stream_json, getattr(self._task_stream_cm, "data", []))
        if self._memory_sampler is not None and self.artifacts.memory_csv is not None:
            try:
                self._memory_sampler.to_pandas(align=True).to_csv(self.artifacts.memory_csv)
            except Exception as mem_exc:
                _write_json(self.artifacts.memory_csv.with_suffix(".error.json"), {"error": repr(mem_exc)})
        if self.statistical_profile and self.client is not None and self._profile_start is not None:
            _write_client_profiles(
                self.client,
                start=self._profile_start,
                stop=profile_stop,
                worker_path=self.artifacts.worker_profile_json,
                scheduler_path=self.artifacts.scheduler_profile_json,
            )
        _write_json(self.artifacts.scheduler_after_json, dask_cluster_snapshot(self.client))
        _write_json(self.artifacts.phase_timings_json, self.timings)
        print(f"Dask profile artifacts: {self.artifacts.output_dir.resolve()}")
        return None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self._ensure_collectors()
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.timings.append(
                {
                    "phase": name,
                    "status": status,
                    "duration_s": round(time.perf_counter() - start, 6),
                }
            )


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _write_client_profiles(
    client: Any,
    *,
    start: float,
    stop: float,
    worker_path: Path | None,
    scheduler_path: Path | None,
) -> None:
    for path, scheduler in ((worker_path, False), (scheduler_path, True)):
        if path is None:
            continue
        try:
            profile = client.profile(start=start, stop=stop, scheduler=scheduler, plot=False)
        except Exception as exc:
            _write_json(path.with_suffix(".error.json"), {"error": repr(exc), "scheduler": scheduler})
        else:
            _write_json(path, profile)


def dask_dashboard_versions() -> dict[str, str | None]:
    """Return package versions that affect distributed dashboard rendering."""
    out: dict[str, str | None] = {}
    for package in ("dask", "distributed", "bokeh"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def _print_dask_dashboard_versions() -> None:
    versions = dask_dashboard_versions()
    print(
        "Dask dashboard package versions: "
        f"dask={versions['dask']}, distributed={versions['distributed']}, bokeh={versions['bokeh']}"
    )
    bokeh_version = versions.get("bokeh")
    if bokeh_version:
        try:
            bokeh_major = int(bokeh_version.split(".", maxsplit=1)[0])
        except ValueError:
            bokeh_major = 0
        if bokeh_major >= 3:
            print(
                "Dashboard /profile may be unreliable with this Bokeh/Dask combination; "
                "use generated Dask profile artifacts if the Profile tab fails."
            )


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
    min_worker_memory_gb: float = 8.0,
    wait_for_workers: bool | int | None = None,
    wait_timeout: str | int | float | None = "10m",
    local_directory: str | None = None,
    memory_target: float | bool = 0.60,
    memory_spill: float | bool = 0.70,
    memory_pause: float | bool = 0.80,
    memory_terminate: float | bool = 0.95,
    interface: str | None = None,
    worker_extra_args: Sequence[str] | None = None,
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
    memory_per_worker_gb = memory_per_node_gb / processes_per_node
    local_directory = local_directory or os.getenv("DASK_LOCAL_DIRECTORY") or os.getenv("SLURM_TMPDIR") or os.getenv("TMPDIR")
    worker_extra_args = list(worker_extra_args or [])
    expected_workers = n_jobs * processes_per_node

    dask_config.set(
        {
            "distributed.worker.memory.target": memory_target,
            "distributed.worker.memory.spill": memory_spill,
            "distributed.worker.memory.pause": memory_pause,
            "distributed.worker.memory.terminate": memory_terminate,
            "array.slicing.split_large_chunks": True,
            "distributed.scheduler.worker-saturation": 0.95,
            "distributed.scheduler.worker-memory-limit": 0.95,
        }
    )

    if shutil.which("sbatch") is None:
        n_workers = min(n_cpu, os.cpu_count() or 4)
        per_worker_gb = max(float(min_worker_memory_gb), float(memory_per_node_gb) / max(1, n_workers))
        memory_limit = f"{min(per_worker_gb, 512):.1f}GB"
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
            dashboard_address=f":{port}" if port else None,
            name=name,
            local_directory=local_directory,
        )
        client = Client(cluster)
        print("sbatch not found; using local Dask cluster.")
        print(f"Workers: {n_workers}, ~{memory_per_node_gb:.1f} GB budget -> {memory_limit} per worker")
        _print_dask_resource_summary(
            jobs=1,
            processes_per_job=n_workers,
            threads_per_process=1,
            memory_per_worker_gb=per_worker_gb,
            local_directory=local_directory,
        )
        if wait_for_workers is True or type(wait_for_workers) is int:
            _wait_for_workers(client, n_workers if wait_for_workers is True else int(wait_for_workers), wait_timeout)
        if port:
            print(f"Local dashboard: http://localhost:{port}")
        _print_dask_dashboard_versions()
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
            f"({memory_per_worker_gb:.2f} GB/worker)."
        )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    scheduler_options: dict[str, Any] = {"dashboard_address": f":{port}"} if port else {}
    if interface:
        scheduler_options["interface"] = interface

    cluster = SLURMCluster(
        name=name,
        cores=n_cpu,
        processes=processes_per_node,
        n_workers=0,
        memory=f"{memory_per_node_gb}GB",
        account=account,
        queue=part,
        walltime=walltime,
        scheduler_options=scheduler_options,
        interface=interface,
        local_directory=local_directory,
        worker_extra_args=worker_extra_args,
        job_extra_directives=[
            f"--output={log_dir}/%j.out",
            f"--error={log_dir}/%j.err",
            "--propagate=STACK",
        ],
        job_script_prologue=prologue,
        python=python,
    )
    cluster.scale(jobs=n_jobs)
    print(cluster.job_script())

    client = Client(cluster)
    should_wait = True if wait_for_workers is None else wait_for_workers
    if should_wait is True or type(should_wait) is int:
        n_wait = expected_workers if should_wait is True else int(should_wait)
        _wait_for_workers(client, n_wait, wait_timeout)
    _print_dask_resource_summary(
        jobs=n_jobs,
        processes_per_job=processes_per_node,
        threads_per_process=max(1, n_cpu // processes_per_node),
        memory_per_worker_gb=memory_per_worker_gb,
        local_directory=local_directory,
    )
    print(f"Scheduler workers: {len(client.scheduler_info().get('workers', {}))}/{expected_workers}")
    if port:
        dashboard_address = cluster.scheduler_address
        host = dashboard_address.split("//")[-1].split(":")[0]
        print(f"Remote dashboard address: http://{host}:{port}")
        print(f"Setup ssh port forwarding: ssh -L {port}:{host}:{port} levante")
        print(f"Local dashboard address: http://localhost:{port}")
    _print_dask_dashboard_versions()
    return cluster, client


def _wait_for_workers(client: Any, n_workers: int, timeout: str | int | float | None) -> None:
    if n_workers <= 0:
        return
    print(f"Waiting for {n_workers} Dask workers (timeout={timeout})...")
    try:
        client.wait_for_workers(n_workers, timeout=timeout)
    except Exception as exc:
        snapshot = dask_cluster_snapshot(client)
        print(f"Dask worker wait incomplete: {exc!r}")
        print(f"Workers currently visible: {snapshot.get('workers', 0)}/{n_workers}")
        raise


def _print_dask_resource_summary(
    *,
    jobs: int,
    processes_per_job: int,
    threads_per_process: int,
    memory_per_worker_gb: float,
    local_directory: str | None,
) -> None:
    print("Dask resource summary:")
    print(f"  - Jobs/nodes: {jobs}")
    print(f"  - Worker processes/job: {processes_per_job}")
    print(f"  - Threads/process: {threads_per_process}")
    print(f"  - Memory/worker: {memory_per_worker_gb:.2f} GB")
    print(f"  - Local directory: {local_directory or 'Dask default'}")
