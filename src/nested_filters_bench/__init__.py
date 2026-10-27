import argparse
import asyncio
import logging
from dataclasses import fields

import numpy as np
from qdrant_client import AsyncQdrantClient

from nested_filters_bench.config import BenchConfig
from nested_filters_bench.corpus import load_corpus, load_query_vectors
from nested_filters_bench.indexing import prepare_collection
from nested_filters_bench.queries import BenchQuery, Difficulty, Mode, sample_queries
from nested_filters_bench.report import print_table, save_json
from nested_filters_bench.runner import CellResult, run_cell

logger = logging.getLogger(__name__)


def parse_args() -> BenchConfig:
    """Turn the fields of `BenchConfig` into command-line options."""
    parser = argparse.ArgumentParser(description="Benchmark Qdrant nested filters.")
    for field in fields(BenchConfig):
        flag = f"--{field.name.replace('_', '-')}"
        if field.type is bool:
            parser.add_argument(flag, action="store_true")
        else:
            parser.add_argument(flag, type=field.type, default=field.default)
    return BenchConfig(**vars(parser.parse_args()))


async def run(config: BenchConfig) -> list[CellResult]:
    """Load data, build the exact answers, index, then measure every mode on every profile."""
    logger.info("Loading %d arXiv papers", config.points)
    corpus = load_corpus(config.points)
    rng = np.random.default_rng(config.seed + 1)

    logger.info("Computing exact ground truth")
    workloads: dict[Difficulty, list[BenchQuery]] = {
        difficulty: sample_queries(
            corpus,
            difficulty,
            load_query_vectors(config.queries, rng),
            config.k,
            rng,
        )
        for difficulty in Difficulty
    }

    client = AsyncQdrantClient(url=config.url, timeout=600)
    for per_element_index in (False, True):
        logger.info("Preparing collection (per-element index: %s)", per_element_index)
        await prepare_collection(client, corpus, config, per_element_index)

    results: list[CellResult] = []
    for difficulty, queries in workloads.items():
        for mode in Mode:
            logger.info("Running %s / %s", difficulty, mode)
            results.append(await run_cell(client, mode, difficulty, queries, config))
    await client.close()
    return results


def main() -> None:
    """Entry point of the `nested-filters-bench` command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = parse_args()
    results = asyncio.run(run(config))
    print_table(results)
    save_json(results, config)
