import asyncio
from collections.abc import Iterator

from qdrant_client import AsyncQdrantClient, models

from nested_filters_bench.config import BenchConfig
from nested_filters_bench.corpus import Corpus, build_payloads


def collection_name(config: BenchConfig, per_element_index: bool) -> str:
    """Two collections hold the same data: one with regular indexes, one with per-element indexes."""
    kind = "indexed" if per_element_index else "regular"
    return f"{config.collection_prefix}_{kind}"


IndexSchema = (
    models.KeywordIndexParams | models.IntegerIndexParams | models.DatetimeIndexParams
)


def _index_schemas(per_element_index: bool) -> dict[str, IndexSchema]:
    """Payload indexes to create. `nested=<array>` makes an index per-element."""

    def nested(array: str) -> str | None:
        return array if per_element_index else None

    return {
        "authors[].last": models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, nested=nested("authors")
        ),
        "authors[].first": models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, nested=nested("authors")
        ),
        "versions[].version": models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=True,
            range=True,
            nested=nested("versions"),
        ),
        "versions[].created": models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME, nested=nested("versions")
        ),
        "categories": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    }


async def create_collection(
    client: AsyncQdrantClient, name: str, dim: int, per_element_index: bool
) -> None:
    """Create an empty collection with its payload indexes (before ingestion, as Qdrant advises)."""
    await client.recreate_collection(
        name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    for field, schema in _index_schemas(per_element_index).items():
        await client.create_payload_index(name, field_name=field, field_schema=schema)


def iter_points(corpus: Corpus, chunk_size: int) -> Iterator[models.PointStruct]:
    """Yield the corpus as points, building payloads one chunk at a time."""
    for start in range(0, len(corpus), chunk_size):
        stop = min(start + chunk_size, len(corpus))
        vectors = corpus.vectors[start:stop].tolist()
        payloads = build_payloads(corpus, start, stop)
        for point_id, vector, payload in zip(
            range(start, stop), vectors, payloads, strict=True
        ):
            yield models.PointStruct(id=point_id, vector=vector, payload=payload)


def ingest(
    client: AsyncQdrantClient, name: str, corpus: Corpus, config: BenchConfig
) -> None:
    """Upload every point; the client handles batching, parallelism and retries."""
    client.upload_points(
        name,
        iter_points(corpus, config.batch_size),
        batch_size=config.batch_size,
        parallel=config.upload_concurrency,
        wait=True,
    )


async def wait_until_green(client: AsyncQdrantClient, name: str) -> None:
    """Wait until Qdrant finishes building the indexes."""
    while (await client.get_collection(name)).status != models.CollectionStatus.GREEN:
        await asyncio.sleep(2.0)


async def is_ready(client: AsyncQdrantClient, name: str, points: int) -> bool:
    """Whether the collection exists and holds exactly `points` points."""
    if not await client.collection_exists(name):
        return False
    return (await client.count(name, exact=True)).count == points


async def prepare_collection(
    client: AsyncQdrantClient,
    corpus: Corpus,
    config: BenchConfig,
    per_element_index: bool,
) -> None:
    """Create, fill and index one collection (skipped with `--reuse` if it is complete)."""
    name = collection_name(config, per_element_index)
    if config.reuse and await is_ready(client, name, len(corpus)):
        return
    await create_collection(client, name, corpus.vectors.shape[1], per_element_index)
    ingest(client, name, corpus, config)
    await wait_until_green(client, name)
