"""Disjoint active-segment accounting for interleaved, in-process searches."""
from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time


class SegmentLedger:
    def __init__(self, counter, path=None, clock=time.monotonic):
        self.counter, self.clock = counter, clock
        self.path = Path(path) if path is not None else None
        self.rows = []
        self.active = False
        self.origin = self._calls()
        self.started = clock()
        self.last = self.origin
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise FileExistsError(self.path)

    def _calls(self):
        n = self.counter()
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError('model counter must be a nonnegative integer')
        return n

    @contextmanager
    def segment(self, stage, pair=None, seed=None, **metadata):
        if self.active:
            raise RuntimeError('nested segments would double-count calls')
        c0, t0 = self._calls(), self.clock()
        if c0 != self.last:
            raise RuntimeError(f'unattributed model calls: {c0 - self.last}')
        self.active = True
        status, error = 'ran', None
        try:
            yield
        except BaseException as exc:
            status, error = 'error', f'{type(exc).__name__}: {exc}'
            raise
        finally:
            c1, elapsed = self._calls(), self.clock() - t0
            self.active = False
            if c1 < c0 or not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError('counter or monotonic clock decreased')
            row = dict(segment=len(self.rows) + 1, stage=stage, pair=pair, seed=seed,
                       calls=c1 - c0, seconds=elapsed, calls_through=c1 - self.origin,
                       status=status, error=error, **metadata)
            self.rows.append(row)
            self.last = c1
            if self.path is not None:
                with self.path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                    f.flush()

    def totals(self, pair=None, seed=None, stage=None):
        rows = [r for r in self.rows if (pair is None or r['pair'] == pair)
                and (seed is None or r['seed'] == seed)
                and (stage is None or r['stage'] == stage)]
        return {'calls': sum(r['calls'] for r in rows),
                'seconds': sum(r['seconds'] for r in rows)}

    def summary(self):
        if self.active or self._calls() != self.last:
            raise RuntimeError('unfinished or unaccounted segment')
        total = self.totals()
        if total['calls'] != self.last - self.origin:
            raise RuntimeError('segment sum differs from model counter')
        stages = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
        for row in self.rows:
            for key in ('calls', 'seconds'):
                stages[row['stage']][key] += row[key]
        return {'schema': 'chavis_ultra_segments_v1', 'total': total,
                'elapsed_seconds': self.clock() - self.started,
                'segments': len(self.rows), 'by_stage': dict(stages),
                'accounting': 'Disjoint active segments. Total includes pending candidates; do not add breakdowns to total.'}
