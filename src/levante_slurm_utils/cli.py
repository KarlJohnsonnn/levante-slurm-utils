"""CLI for Levante Slurm resource sizing."""

from __future__ import annotations

import argparse
import json
import sys

from .compute import scaling_plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Levante Dask/Slurm resource recommendation.")
    parser.add_argument("--time-steps", type=int, required=True, help="Number of time steps.")
    parser.add_argument("--experiments", type=int, required=True, help="Number of experiments.")
    parser.add_argument("--stations", type=int, required=True, help="Number of stations or locations.")
    parser.add_argument("--debug", action="store_true", help="Return small debug allocation.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = scaling_plan(args.time_steps, args.experiments, args.stations, args.debug)
    payload = {
        "n_nodes": plan.n_nodes,
        "n_cpu": plan.n_cpu,
        "memory_gb": plan.memory_gb,
        "n_workers": plan.n_workers,
        "walltime": plan.walltime,
        "total_workload": plan.total_workload,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"total_workload={plan.total_workload}")
        print(f"n_nodes={plan.n_nodes}")
        print(f"n_cpu={plan.n_cpu}")
        print(f"memory_gb={plan.memory_gb}")
        print(f"n_workers={plan.n_workers}")
        print(f"walltime={plan.walltime}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
