"""Re-identification evaluation and threshold selection.

Given labelled crops (identity -> list of embeddings) this computes the
standard ReID metrics (rank-1 / rank-5 / mAP) plus what operations actually
needs: the similarity thresholds that hit a target false-accept rate, so the
`match_threshold` / `review_threshold` in the store config are chosen from
data rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReIDMetrics:
    rank1: float
    rank5: float
    map: float
    n_queries: int
    n_gallery: int

    def as_dict(self) -> dict:
        return {
            "rank1": round(self.rank1, 4),
            "rank5": round(self.rank5, 4),
            "mAP": round(self.map, 4),
            "queries": self.n_queries,
            "gallery": self.n_gallery,
        }


def evaluate_reid(
    query: np.ndarray, query_ids: np.ndarray, gallery: np.ndarray, gallery_ids: np.ndarray
) -> ReIDMetrics:
    sims = query @ gallery.T  # embeddings are L2-normalised
    order = np.argsort(-sims, axis=1)
    ranked_ids = gallery_ids[order]
    hits = ranked_ids == query_ids[:, None]

    rank1 = float(hits[:, 0].mean())
    rank5 = float(hits[:, :5].any(axis=1).mean())
    aps = []
    for row in hits:
        if not row.any():
            aps.append(0.0)
            continue
        idx = np.where(row)[0]
        precision_at_k = np.arange(1, len(idx) + 1) / (idx + 1)
        aps.append(float(precision_at_k.mean()))
    return ReIDMetrics(rank1, rank5, float(np.mean(aps)), len(query), len(gallery))


def similarity_distributions(
    embeddings: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All pairwise similarities split into genuine (same id) and impostor (different id)."""
    sims = embeddings @ embeddings.T
    same = ids[:, None] == ids[None, :]
    iu = np.triu_indices(len(ids), k=1)
    genuine = sims[iu][same[iu]]
    impostor = sims[iu][~same[iu]]
    return genuine, impostor


def choose_thresholds(
    embeddings: np.ndarray,
    ids: np.ndarray,
    target_far: float = 0.01,
    review_far: float = 0.05,
) -> dict:
    """Pick match/review thresholds from the impostor distribution.

    match_threshold  -> similarity above which only `target_far` of impostor pairs fall
    review_threshold -> same for `review_far`; the band in between is sent to review
    Also reports the genuine-accept rate you get at each, so the trade-off is explicit.
    """
    genuine, impostor = similarity_distributions(embeddings, ids)
    if len(impostor) == 0 or len(genuine) == 0:
        raise ValueError("need at least two identities with two samples each")
    match_t = float(np.quantile(impostor, 1 - target_far))
    review_t = float(np.quantile(impostor, 1 - review_far))
    return {
        "match_threshold": round(match_t, 4),
        "review_threshold": round(review_t, 4),
        "genuine_accept_at_match": round(float((genuine >= match_t).mean()), 4),
        "genuine_accept_at_review": round(float((genuine >= review_t).mean()), 4),
        "genuine_pairs": int(len(genuine)),
        "impostor_pairs": int(len(impostor)),
        "genuine_mean": round(float(genuine.mean()), 4),
        "impostor_mean": round(float(impostor.mean()), 4),
    }
