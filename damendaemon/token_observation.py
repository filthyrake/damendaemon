"""Constant-memory host token observations, not isolated engine/kernel timing."""
from dataclasses import dataclass


@dataclass
class TokenObservation:
    count: int = 0
    first: float | None = None
    last: float | None = None
    max_gap: float = 0.0
    sequential: bool = True

    def observe(self, offset, *, sequential=True):
        # Offsets come from the owning runtime's monotonic clock. Speculative
        # bursts have no trustworthy per-token completion timestamps.
        self.sequential = self.sequential and sequential
        if self.last is not None:
            self.max_gap = max(self.max_gap, offset - self.last)
        else:
            self.first = offset
        self.last = offset
        self.count += 1

    def report(self):
        interval = self.last - self.first if self.count > 1 else None
        gaps = self.sequential and interval is not None
        return dict(scope='server_sample_observation', sampled_tokens=self.count,
                    first_sample_seconds=self.first, last_sample_seconds=self.last,
                    sequential_token_timing=self.sequential,
                    mean_gap_seconds=interval / (self.count - 1) if gaps else None,
                    max_gap_seconds=self.max_gap if gaps else None,
                    interval_tokens_per_second=(self.count - 1) / interval
                    if gaps and interval > 0 else None)
