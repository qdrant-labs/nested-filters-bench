# nested-filters-bench

Benchmarks Qdrant 1.20 nested object filters against flat filters.

## Setup

```bash
docker run -d -p 6333:6333 qdrant/qdrant:v1.20.0
uv sync
```

## Run

```bash
uv run nested-filters-bench --points 1000000 --queries 200
```

Use `--reuse` to skip ingestion when the collections already exist. Run `--help` for all options.

## Dataset

[arXiv metadata with gte-base-en-v1.5 abstract embeddings](https://huggingface.co/datasets/bluuebunny/arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5) (768 dimensions, cosine, up to 1.2M papers). `--points` takes the newest yearly files first (2023 and older). Query vectors are abstracts of 2024 papers, so they are not in the corpus.

Payload per paper:

- `authors`: `[{last, first}]`, reshaped from `authors_parsed`
- `versions`: `[{version, created}]`, with `created` as a datetime
- `categories`: list of keywords

The first run downloads the parquet files into the Hugging Face cache (about 0.8 GB per year).

## Queries

| Profile | Filter |
| --- | --- |
| `versions` | a revision (v2 or later) created in a 90-day window |
| `author` | an author with a given last and first name |
| `author_versions` | the author filter, plus a version created in a 365-day window |
| `versions_category` | a revision created in a 365-day window, plus a category |

Filter values are sampled from the data. Each filter must match at least k papers.

## Modes

| Mode | Filter | Indexes |
| --- | --- | --- |
| `flat` | `authors[].field`, `versions[].field` conditions | regular |
| `nested_regular` | `nested` condition | regular |
| `nested_indexed` | `nested` condition | per-element (`nested="authors"`, `nested="versions"`) |

A flat filter matches conditions across different array elements (for example, a first name from one author and a last name from another), so it returns different points than a nested filter. Every mode is scored against the same exact brute-force answers under nested semantics, so flat loses recall when it returns the wrong papers.

## Metrics

Latency (p50/p95/p99, sequential), throughput (QPS at `--concurrency`), and recall, MRR and nDCG at k=10. Relevant points are the exact top-k under the filter. 
