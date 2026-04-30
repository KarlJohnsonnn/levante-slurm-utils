# Levante Slurm Utils

Small Python package for sizing Dask workloads and starting local or
Slurm-backed Dask clusters on DKRZ Levante-style systems.


## HPC Resource Usage Philosophy

**Use only as much resources as the data actually require.**

1. Start with the smallest faithful version of the problem (reduced domain/time/diagnostics).
2. Keep data lazy and chunked until you have reduced it.
3. Compute only what you need for the next step (e.g. a histogram, not the full raw volume).
4. Measure bottlenecks first; optimize second.
5. Increase resources only after measurement shows a real need.

In short: **scale only by evidence, not by guessing**.

### Outline

| Term | Meaning |
|---|---|
| **Data movement** | Copying large arrays between disk, RAM, workers, or nodes. Often slower than math itself. |
| **Lazy array** | Data is not loaded yet; Python stores a plan of operations first. |
| **Chunking** | Splitting big arrays into smaller blocks so processing fits memory. |
| **Materialize** | Actually load/compute the array now (`.compute()` or `.values`). |
| **Scale up** | Request more CPUs, RAM, or nodes. |
| **Profile** | Measure where runtime is spent (CPU, memory, I/O, scheduler overhead). |

### Before requesting more memory or workers check the following:

1. Increase stride / downsampling.
2. Reduce histogram bin count (e.g. `96 -> 64 -> 48`).
3. Process one model run at a time.
4. Cache reduced intermediates to NetCDF/Zarr and reuse them.
5. Switch to `float32` where scientific accuracy allows.
6. Scale cluster resources only if full-resolution output is required.

## Install

```bash
python -m pip install -e '.[distributed,test]'
```

Core chunk-sizing helpers need `dask` and `xarray`. Starting clusters needs
`dask[distributed]`; Slurm mode also needs `dask-jobqueue` and `sbatch` on
`PATH`.

## Resource Sizing

The sizing heuristic is intentionally simple and should be treated as a starting point.

```bash
levante-slurm-size --time-steps 300 --experiments 8 --stations 3
levante-slurm-size --time-steps 1200 --experiments 80 --stations 5 --json
```

Python API:

```python
from levante_slurm_utils import calculate_optimal_scaling

n_nodes, n_cpu, mem_gb, n_workers, walltime = calculate_optimal_scaling(
    n_time_steps=300,
    n_experiments=8,
    n_stations=3,
)
```


## Dask Cluster

Local fallback works on laptops and CI when `sbatch` is absent:

```python
from levante_slurm_utils import allocate_resources

cluster, client = allocate_resources(n_cpu=8, n_jobs=1, m=16, port=None)
```

On Levante:

```python
cluster, client = allocate_resources(
    n_cpu=128,
    n_jobs=2,
    m=128,
    walltime="06:00:00",
    part="compute",
    account="bb1234",
    conda_env="my_env",
    python="/home/b/<user>/.conda/envs/my_env/bin/python",
    log_dir="./logs",
)
cluster.scale(4)
```

Set `account`, `conda_env`, and `python` for your project/user.


## Chunking Helpers

```python
from levante_slurm_utils import auto_chunk_dataset, describe_chunk_plan

ds, chunks = auto_chunk_dataset(ds, min_chunk_mb=64, max_chunk_mb=512)
print(describe_chunk_plan(ds, chunks))
```

`auto_chunk_dataset` uses active Dask worker memory if a client exists,
otherwise local available memory.

## Tests

```bash
python -m pytest
levante-slurm-size --help
```

