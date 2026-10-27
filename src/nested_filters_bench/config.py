from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    """Command-line options of the benchmark."""

    url: str = "http://localhost:6333"  # Qdrant server
    points: int = 200_000  # papers to load and index
    queries: int = 200  # queries per profile
    k: int = 10  # results per query (recall, MRR and nDCG are computed @k)
    concurrency: int = 16  # parallel requests in the throughput test
    hnsw_ef: int = 128  # HNSW search breadth; higher is slower but more exact
    batch_size: int = 1_000  # points per upload batch
    upload_concurrency: int = 4  # parallel upload workers
    warmup: int = 20  # queries run (and discarded) before timing starts
    seed: int = 42  # random seed for query sampling
    collection_prefix: str = "nested_bench"  # prefix of the two collection names
    output: str = "results.json"  # where the results are saved
    reuse: bool = False  # skip ingestion if the collections already exist
