from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from qdrant_client import models

from nested_filters_bench.corpus import Corpus

DAY_SECONDS = 86_400
MAX_ATTEMPTS = 100_000  # filter draws per query before giving up


class Mode(StrEnum):
    """The three ways to run a filter that are compared."""

    FLAT = "flat"  # `array[].key` conditions: each may match a different element
    NESTED_REGULAR = (
        "nested_regular"  # `nested` filter, regular indexes: payloads are read
    )
    NESTED_INDEXED = "nested_indexed"  # `nested` filter, per-element indexes

    @property
    def nested(self) -> bool:
        """Whether the filter uses a `nested` condition."""
        return self is not Mode.FLAT

    @property
    def per_element_index(self) -> bool:
        """Whether the collection has per-element payload indexes."""
        return self is Mode.NESTED_INDEXED


class Difficulty(StrEnum):
    """Query profiles: which kinds of filter conditions a query uses."""

    VERSIONS = "versions"
    AUTHOR = "author"
    AUTHOR_VERSIONS = "author_versions"
    VERSIONS_CATEGORY = "versions_category"


@dataclass(frozen=True)
class Profile:
    """Recipe for random filters of one profile."""

    author: bool  # filter on an author's last and first name
    min_version: (
        int | None
    )  # filter on a version at least this number (None: no version filter)
    window_days: int  # length of the version's creation-date window
    category: bool  # filter on a category


# Profile(author, min_version, window_days, category)
PROFILES: dict[Difficulty, Profile] = {
    Difficulty.VERSIONS: Profile(False, 2, 90, False),
    Difficulty.AUTHOR: Profile(True, None, 0, False),
    Difficulty.AUTHOR_VERSIONS: Profile(True, 1, 365, False),
    Difficulty.VERSIONS_CATEGORY: Profile(False, 2, 365, True),
}


@dataclass(frozen=True)
class AuthorSpec:
    """One author: both names must match on the same author."""

    last: str
    first: str


@dataclass(frozen=True)
class VersionSpec:
    """A version numbered at least `min_version`, created within the window."""

    min_version: int
    created_from: datetime
    created_to: datetime


@dataclass(frozen=True)
class QuerySpec:
    """A concrete filter. A field set to None is not used."""

    author: AuthorSpec | None
    version: VersionSpec | None
    category: str | None


@dataclass(frozen=True)
class BenchQuery:
    """A query vector, its filter, and the exact answers to compare against."""

    vector: NDArray[np.float32]
    spec: QuerySpec
    truth: list[int]  # exact top-k ids among the papers matching the nested filter
    selectivity: float  # share of all papers that match the nested filter (0.01 = 1%)


def build_filter(mode: Mode, spec: QuerySpec) -> models.Filter:
    """Build the Qdrant filter: one nested condition per array, or flat `array[].key` conditions."""
    elements: dict[str, list[models.FieldCondition]] = {}
    if spec.author:
        elements["authors"] = [
            models.FieldCondition(
                key="last", match=models.MatchValue(value=spec.author.last)
            ),
            models.FieldCondition(
                key="first", match=models.MatchValue(value=spec.author.first)
            ),
        ]
    if spec.version:
        elements["versions"] = [
            models.FieldCondition(
                key="version", range=models.Range(gte=spec.version.min_version)
            ),
            models.FieldCondition(
                key="created",
                range=models.DatetimeRange(
                    gte=spec.version.created_from, lte=spec.version.created_to
                ),
            ),
        ]

    must: list[models.Condition] = []
    for array, conditions in elements.items():
        if mode.nested:
            # Keys are relative to the array element.
            nested = models.Nested(
                key=array, filter=models.Filter(must=list(conditions))
            )
            must.append(models.NestedCondition(nested=nested))
        else:
            must += [
                c.model_copy(update={"key": f"{array}[].{c.key}"}) for c in conditions
            ]
    if spec.category:
        must.append(
            models.FieldCondition(
                key="categories", match=models.MatchAny(any=[spec.category])
            )
        )
    return models.Filter(must=must)


def match_mask(corpus: Corpus, spec: QuerySpec) -> NDArray[np.bool_]:
    """Boolean mask over papers that satisfy the filter, computed with numpy.

    Nested semantics: one array element must meet all conditions of its array.
    """
    authors, versions, categories = corpus.authors, corpus.versions, corpus.categories

    # Per array: the owner paper of every element, and one element-level mask per condition.
    groups: list[tuple[NDArray[np.int64], list[NDArray[np.bool_]]]] = []
    if spec.author:
        groups.append(
            (
                authors.point,
                [
                    authors.last == authors.last_vocab.ids[spec.author.last],
                    authors.first == authors.first_vocab.ids[spec.author.first],
                ],
            )
        )
    if spec.version:
        start = int(spec.version.created_from.timestamp())
        end = int(spec.version.created_to.timestamp())
        groups.append(
            (
                versions.point,
                [
                    versions.number >= spec.version.min_version,
                    (versions.created >= start) & (versions.created <= end),
                ],
            )
        )
    if spec.category:
        code = categories.vocab.ids[spec.category]
        groups.append((categories.point, [categories.code == code]))

    mask = np.ones(len(corpus), dtype=np.bool_)
    for point, conditions in groups:
        # An element passes if it meets all conditions of its array.
        element = np.logical_and.reduce(conditions)
        # A paper passes if any of its elements passes.
        owners = np.zeros(len(corpus), dtype=np.bool_)
        owners[point[element]] = True
        mask &= owners
    return mask


def exact_top_k(
    vectors: NDArray[np.float32],
    query: NDArray[np.float32],
    mask: NDArray[np.bool_],
    k: int,
) -> list[int]:
    """Brute-force top-k among the masked points (cosine on unit vectors)."""
    # Score all points, then rule out the ones that fail the filter.
    scores = np.where(mask, vectors @ query, -np.inf)
    # Pick the k best without a full sort, then order those k by score.
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])].tolist()


def _sample_spec(
    corpus: Corpus,
    profile: Profile,
    bounds: tuple[int, int],
    rng: np.random.Generator,
) -> QuerySpec:
    """Draw a random filter for the profile, using values that exist in the corpus."""
    author = None
    if profile.author:
        # A random author row picks common names more often.
        row = rng.integers(len(corpus.authors.point))
        author = AuthorSpec(
            corpus.authors.last_vocab.names[corpus.authors.last[row]],
            corpus.authors.first_vocab.names[corpus.authors.first[row]],
        )

    version = None
    if profile.min_version is not None:
        # A window starting at a random time within the corpus' creation dates.
        low, high = bounds
        window = profile.window_days * DAY_SECONDS
        start = datetime.fromtimestamp(
            int(rng.integers(low, max(low + 1, high - window))), UTC
        )
        version = VersionSpec(
            profile.min_version, start, start + timedelta(seconds=window)
        )

    category = None
    if profile.category:
        row = rng.integers(len(corpus.categories.point))
        category = corpus.categories.vocab.names[corpus.categories.code[row]]

    return QuerySpec(author, version, category)


def sample_queries(
    corpus: Corpus,
    difficulty: Difficulty,
    vectors: NDArray[np.float32],
    k: int,
    rng: np.random.Generator,
) -> list[BenchQuery]:
    """Pair each query vector with a random filter and its exact top-k answers."""
    profile = PROFILES[difficulty]
    created = corpus.versions.created
    bounds = (int(created.min()), int(created.max()))

    queries: list[BenchQuery] = []
    for vector in vectors:
        # Redraw the filter until it matches at least k papers.
        for _ in range(MAX_ATTEMPTS):
            spec = _sample_spec(corpus, profile, bounds, rng)
            mask = match_mask(corpus, spec)
            if mask.sum() >= k:
                break
        else:
            raise RuntimeError(f"No filter with {k}+ matches found; use more points")
        queries.append(
            BenchQuery(
                vector=vector,
                spec=spec,
                truth=exact_top_k(corpus.vectors, vector, mask, k),
                selectivity=float(mask.mean()),
            )
        )
    return queries
