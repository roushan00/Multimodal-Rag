from collections import defaultdict
from typing import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60,
        weights: Sequence[float] | None = None) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1.

    Uses only ranks, so it needs no score calibration between a cosine score (Path A)
    and a MaxSim score (Path B), which live on completely different scales.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[Hashable, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def dedupe_keep_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
