import asyncio
import time
from dataclasses import dataclass

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from nested_filters_bench.config import BenchConfig
from nested_filters_bench.indexing import collection_name
from nested_filters_bench.metrics import Quality, mean_quality
from nested_filters_bench.queries import BenchQuery, Difficulty, Mode, build_filter


@dataclass(frozen=True)
class LatencyMs:
    """Client-side latency of one search, in milliseconds."""

    mean: float
    p50: float
    p95: float
    p99: float


@dataclass(frozen=True)
class CellResult:
    """Results of one mode on one query profile."""

    mode: Mode
    difficulty: Difficulty
    selectivity: float  # average share of papers matching the filters
    latency: LatencyMs  # measured with one request at a time
    qps: float  # queries per second with `concurrency` requests at a time
    quality: Quality  # against the exact nested-filter answers


@dataclass(frozen=True)
class Answer:
    """Result of one search."""

    ids: list[int]
    seconds: float


async def _search(
    client: AsyncQdrantClient, mode: Mode, query: BenchQuery, config: BenchConfig
) -> Answer:
    """Run one filtered search and time it."""
    start = time.perf_counter()
    response = await client.query_points(
        collection_name(config, mode.per_element_index),
        query=query.vector.tolist(),
        query_filter=build_filter(mode, query.spec),
        limit=config.k,
        search_params=models.SearchParams(hnsw_ef=config.hnsw_ef),
    )
    return Answer([int(p.id) for p in response.points], time.perf_counter() - start)


async def measure_latency(
    client: AsyncQdrantClient,
    mode: Mode,
    queries: list[BenchQuery],
    config: BenchConfig,
) -> list[Answer]:
    """Run the queries one by one, after a warm-up."""
    for query in queries[: config.warmup]:
        await _search(client, mode, query, config)
    return [await _search(client, mode, query, config) for query in queries]


async def measure_qps(
    client: AsyncQdrantClient,
    mode: Mode,
    queries: list[BenchQuery],
    config: BenchConfig,
) -> float:
    """Run all queries with limited concurrency and return queries per second."""
    semaphore = asyncio.Semaphore(config.concurrency)

    async def bounded(query: BenchQuery) -> Answer:
        async with semaphore:
            return await _search(client, mode, query, config)

    start = time.perf_counter()
    await asyncio.gather(*(bounded(q) for q in queries))
    return len(queries) / (time.perf_counter() - start)


def summarize_latency(answers: list[Answer]) -> LatencyMs:
    """Mean and percentiles of the measured times."""
    ms = np.array([a.seconds for a in answers]) * 1000
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return LatencyMs(float(ms.mean()), float(p50), float(p95), float(p99))


async def run_cell(
    client: AsyncQdrantClient,
    mode: Mode,
    difficulty: Difficulty,
    queries: list[BenchQuery],
    config: BenchConfig,
) -> CellResult:
    """Measure latency, throughput and quality of one mode on one profile."""
    answers = await measure_latency(client, mode, queries, config)
    qps = await measure_qps(client, mode, queries, config)
    retrieved = [a.ids for a in answers]
    return CellResult(
        mode=mode,
        difficulty=difficulty,
        selectivity=float(np.mean([q.selectivity for q in queries])),
        latency=summarize_latency(answers),
        qps=qps,
        quality=mean_quality(retrieved, [q.truth for q in queries], config.k),
    )
