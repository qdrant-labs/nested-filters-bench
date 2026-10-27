from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

# Hugging Face dataset: one parquet file per year, with abstract embeddings.
REPO = "bluuebunny/arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5"
QUERY_YEAR = 2024  # query vectors come from this year, so they are not in the corpus
CORPUS_YEARS = range(2023, 2006, -1)  # files are read newest first
DATETIME_FORMAT = "%a, %d %b %Y %H:%M:%S GMT"  # format of a version's `created`
COLUMNS = ["id", "title", "license", "categories", "versions", "authors_parsed"]
EMBEDDING = "abstract_embedding"


@dataclass(frozen=True)
class Vocabulary:
    """Maps strings to integer codes and back."""

    names: list[str]
    ids: dict[str, int]

    @classmethod
    def encode(cls, array: pa.Array) -> tuple[NDArray[np.int32], Vocabulary]:
        """Replace each string by an integer code; also return the code book."""
        encoded = array.dictionary_encode()
        names = encoded.dictionary.to_pylist()
        ids = {name: i for i, name in enumerate(names)}
        return encoded.indices.to_numpy(), cls(names, ids)


@dataclass(frozen=True)
class Authors:
    """One row per author of each paper."""

    point: NDArray[np.int64]  # row number of the paper that owns this author
    last: NDArray[np.int32]  # last name, as a code from `last_vocab`
    first: NDArray[np.int32]  # first name, as a code from `first_vocab`
    last_vocab: Vocabulary
    first_vocab: Vocabulary


@dataclass(frozen=True)
class Versions:
    """One row per version of each paper; `created` is in epoch seconds."""

    point: NDArray[np.int64]  # row number of the paper that owns this version
    number: NDArray[np.int64]  # 1 for "v1", 2 for "v2", ...
    created: NDArray[np.int64]


@dataclass(frozen=True)
class Categories:
    """One row per category of each paper."""

    point: NDArray[np.int64]  # row number of the paper that owns this category
    code: NDArray[np.int32]  # category name, as a code from `vocab`
    vocab: Vocabulary


@dataclass(frozen=True)
class Corpus:
    """arXiv papers: normalized abstract embeddings, payload source and element-wise metadata."""

    vectors: NDArray[np.float32]  # (papers, 768), unit length
    table: pa.Table  # metadata per paper, used to build payloads
    authors: Authors  # the three below hold the same metadata as flat numpy rows
    versions: Versions
    categories: Categories

    def __len__(self) -> int:
        return len(self.vectors)


def _read_year(year: int) -> tuple[pa.Table, NDArray[np.float32]]:
    """Read one yearly file: the metadata table and the embeddings matrix (papers x 768)."""
    path = hf_hub_download(REPO, f"{year}.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=[*COLUMNS, EMBEDDING])

    # A list<float> column becomes one flat buffer; reshape it to one row per paper.
    vectors = table[EMBEDDING].combine_chunks().values.to_numpy()
    vectors = vectors.reshape(table.num_rows, -1)

    # "cs.AI cs.LG" -> ["cs.AI", "cs.LG"]
    categories = pc.split_pattern(table["categories"], " ")

    # Rebuild each paper's list of versions, with `created` parsed into a timestamp.
    versions = table["versions"].combine_chunks()
    flat = versions.flatten()
    created = pc.strptime(flat.field("created"), DATETIME_FORMAT, "s")
    structs = pa.StructArray.from_arrays(
        [flat.field("version"), created], names=["version", "created"]
    )
    versions = pa.ListArray.from_arrays(versions.offsets, structs)

    table = table.drop_columns([EMBEDDING, "categories", "versions"])
    table = table.append_column("categories", categories)
    table = table.append_column("versions", versions)
    return table, vectors


def _explode_metadata(table: pa.Table) -> tuple[Authors, Versions, Categories]:
    """Flatten each array into one row per element, tagged with its paper's row number.

    These flat arrays let the ground truth filter with plain numpy comparisons.
    """
    authors = table["authors_parsed"].combine_chunks()
    versions = table["versions"].combine_chunks()
    categories = table["categories"].combine_chunks()

    # [paper 0 has 2 items, paper 1 has 1] -> [0, 0, 1]
    author_point = np.repeat(
        np.arange(len(authors)), pc.list_value_length(authors).to_numpy()
    )
    version_point = np.repeat(
        np.arange(len(versions)), pc.list_value_length(versions).to_numpy()
    )
    category_point = np.repeat(
        np.arange(len(categories)), pc.list_value_length(categories).to_numpy()
    )

    # Each author is a list [last, first, ...]; encode the first two items as integer codes.
    names = authors.flatten()
    last, last_vocab = Vocabulary.encode(pc.list_element(names, 0))
    first, first_vocab = Vocabulary.encode(pc.list_element(names, 1))

    # "v3" -> 3; created timestamps -> epoch seconds.
    flat_versions = versions.flatten()
    number = pc.utf8_slice_codeunits(flat_versions.field("version"), 1).cast(pa.int64())
    created = flat_versions.field("created").cast(pa.int64())

    code, vocab = Vocabulary.encode(categories.flatten())
    return (
        Authors(author_point, last, first, last_vocab, first_vocab),
        Versions(
            version_point,
            number.to_numpy(zero_copy_only=False),
            created.to_numpy(zero_copy_only=False),
        ),
        Categories(category_point, code, vocab),
    )


def load_corpus(points: int) -> Corpus:
    """Download the newest yearly files until `points` papers are collected."""
    tables: list[pa.Table] = []
    chunks: list[NDArray[np.float32]] = []
    total = 0
    for year in CORPUS_YEARS:
        if total >= points:
            break
        table, vectors = _read_year(year)
        tables.append(table)
        chunks.append(vectors)
        total += table.num_rows
    table = pa.concat_tables(tables).slice(0, points).combine_chunks()

    # Unit-length vectors make a dot product equal to cosine similarity.
    vectors = np.concatenate(chunks)[:points]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    authors, versions, categories = _explode_metadata(table)
    return Corpus(vectors, table, authors, versions, categories)


def load_query_vectors(count: int, rng: np.random.Generator) -> NDArray[np.float32]:
    """Embed-ready queries: abstracts of papers outside the corpus."""
    path = hf_hub_download(REPO, f"{QUERY_YEAR}.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=[EMBEDDING])
    vectors = table[EMBEDDING].combine_chunks().values.to_numpy()
    vectors = vectors.reshape(table.num_rows, -1)[
        rng.choice(table.num_rows, count, replace=False)
    ]
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_payloads(corpus: Corpus, start: int, stop: int) -> list[dict[str, Any]]:
    """Build the JSON payloads for points `start` to `stop`."""
    rows = corpus.table.slice(start, stop - start).to_pylist()
    return [
        {
            "title": row["title"],
            "license": row["license"],
            "categories": row["categories"],
            "authors": [
                {"last": author[0], "first": author[1]}
                for author in row["authors_parsed"]
            ],
            "versions": [
                {
                    "version": int(v["version"][1:]),
                    "created": v["created"].isoformat() + "Z",
                }
                for v in row["versions"]
            ],
        }
        for row in rows
    ]
