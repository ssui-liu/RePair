"""Exact full-ranking similarities with bounded score-matrix memory."""

import torch


class SimilarityRows:
    """Tensor-like row access, caching only one query chunk at a time.

    Every row still contains every candidate. Chunking changes memory use,
    not retrieval-pool membership or metric definitions.
    """

    def __init__(self, queries, candidates, chunk_size=256):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.queries = queries
        self.candidates = candidates
        self.chunk_size = chunk_size
        self.shape = (queries.shape[0], candidates.shape[0])
        self._start = -1
        self._scores = None

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        start = index // self.chunk_size * self.chunk_size
        if self._start != start:
            with torch.no_grad():
                self._scores = self.queries[start:start + self.chunk_size] @ self.candidates.T
            self._start = start
        return self._scores[index - start]
