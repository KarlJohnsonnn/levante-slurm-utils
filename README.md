# Levante Slurm Utils

`levante-slurm-utils` sizes Dask workloads and starts local or Slurm-backed
Dask clusters on DKRZ Levante-style systems.

It exists to move memory-heavy NetCDF analysis out of fragile notebook kernels
and into reproducible batch or distributed workflows. This should help teams
process larger files with measured resources instead of repeated kernel deaths.

## Contents

- [Install](#install)
- [Test run](#test-run)
- [my_run: dry run with own data](#my_run-dry-run-with-own-data)
- [Configuration](#configuration)
  - [Data layout](#data-layout)
  - [Resource sizing](#resource-sizing)
  - [Dask cluster](#dask-cluster)
  - [Chunking helpers](#chunking-helpers)
- [Reference](#reference)
  - [Resource-use philosophy](#resource-use-philosophy)

## Install

```bash
python -m pip install -e '.[distributed,test]'
```

Core chunk-sizing helpers need `dask` and `xarray`. Starting clusters needs
`dask[distributed]`; Slurm mode also needs `dask-jobqueue` and `sbatch` on
`PATH`.

## Test run

```bash
python -m pytest
levante-slurm-size --help
```

## my_run: dry run with own data

Start with a large NetCDF or Zarr file that usually kills a notebook kernel.
Open it lazily, inspect a chunk plan, and compute only a small preview before
scaling up:

```python
import xarray as xr
from levante_slurm_utils import auto_chunk_dataset, describe_chunk_plan

ds = xr.open_dataset("/path/to/my_run.nc", chunks={})
ds, chunks = auto_chunk_dataset(ds, min_chunk_mb=64, max_chunk_mb=512)
print(describe_chunk_plan(ds, chunks))

preview = ds.isel(time=slice(0, 24)).mean("time").compute()
```

If the local preview is still too large, run the same workflow with a Slurm Dask
cluster and project-specific `account`, `conda_env`, and `python` settings.

## Configuration

### Data layout

Input data should stay on a shared filesystem visible to the notebook, login
node, and Slurm workers. Keep intermediate subsets in NetCDF or Zarr so repeated
analysis does not reload the full raw dataset.

### Resource sizing

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

### Dask cluster

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

### Chunking helpers

```python
from levante_slurm_utils import auto_chunk_dataset, describe_chunk_plan

ds, chunks = auto_chunk_dataset(ds, min_chunk_mb=64, max_chunk_mb=512)
print(describe_chunk_plan(ds, chunks))
```

`auto_chunk_dataset` uses active Dask worker memory if a client exists,
otherwise local available memory.

## Reference

### Resource-use philosophy

Use only as many resources as the data actually require.

1. Start with the smallest faithful version of the problem.
2. Keep data lazy and chunked until you have reduced it.
3. Compute only what you need for the next step.
4. Measure bottlenecks first; optimize second.
5. Increase resources only after measurement shows a real need.

In short: scale only by evidence, not by guessing.

| Term | Meaning |
|---|---|
| **Data movement** | Copying large arrays between disk, RAM, workers, or nodes. Often slower than math itself. |
| **Lazy array** | Data is not loaded yet; Python stores a plan of operations first. |
| **Chunking** | Splitting big arrays into smaller blocks so processing fits memory. |
| **Materialize** | Actually load or compute the array now (`.compute()` or `.values`). |
| **Scale up** | Request more CPUs, RAM, or nodes. |
| **Profile** | Measure where runtime is spent: CPU, memory, I/O, or scheduler overhead. |

Before requesting more memory or workers:

1. Increase stride / downsampling (temporal, spatial, bin, ...) dimensions.
2. Reduce histogram bin count (e.g. `96 -> 64 -> 48`).
3. Process one model run at a time.
4. Create reduced intermediates (e.g. subset of variables) to NetCDF/Zarr and reuse them.
5. Switch to `float32` where scientific accuracy allows.
6. Scale cluster resources only if full-resolution output is required.
