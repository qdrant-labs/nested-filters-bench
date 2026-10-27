import json
from dataclasses import asdict

from nested_filters_bench.config import BenchConfig
from nested_filters_bench.runner import CellResult

# select. = share of papers matching the nested filter. Quality is against the exact nested answers.
HEADER = (
    f"{'difficulty':<18} {'mode':<15} {'select.':>8} {'p50 ms':>8} {'p95 ms':>8} "
    f"{'p99 ms':>8} {'QPS':>8} {'recall':>7} {'MRR':>6} {'nDCG':>6}"
)


def format_row(r: CellResult) -> str:
    """One table line for a result."""
    q = r.quality
    return (
        f"{r.difficulty:<18} {r.mode:<15} {r.selectivity:>8.4%} "
        f"{r.latency.p50:>8.2f} {r.latency.p95:>8.2f} {r.latency.p99:>8.2f} "
        f"{r.qps:>8.1f} {q.recall:>7.3f} {q.mrr:>6.3f} {q.ndcg:>6.3f}"
    )


def print_table(results: list[CellResult]) -> None:
    """Print all results, grouped by profile."""
    print(HEADER)
    for result in sorted(results, key=lambda r: (r.difficulty, r.mode)):
        print(format_row(result))


def save_json(results: list[CellResult], config: BenchConfig) -> None:
    """Write the config and all results to the output file."""
    payload = {"config": asdict(config), "results": [asdict(r) for r in results]}
    with open(config.output, "w") as f:
        json.dump(payload, f, indent=2)
