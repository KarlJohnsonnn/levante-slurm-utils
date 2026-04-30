"""Levante-oriented Dask and Slurm helper utilities."""

from .compute import (
    allocate_resources,
    auto_chunk_dataset,
    calculate_optimal_scaling,
    describe_chunk_plan,
    in_slurm_allocation,
    is_server,
    recommend_target_chunk_mb,
)

__all__ = [
    "allocate_resources",
    "auto_chunk_dataset",
    "calculate_optimal_scaling",
    "describe_chunk_plan",
    "in_slurm_allocation",
    "is_server",
    "recommend_target_chunk_mb",
]
