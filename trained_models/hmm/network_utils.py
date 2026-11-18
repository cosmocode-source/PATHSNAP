"""Route-free directed road graph and bounded, cached shortest paths."""
from collections import OrderedDict
import heapq
import math
import numpy as np


class RoadGraph:
    """Keep adjacency compact and bound the memory used by path searches."""
    def __init__(self, network, cache_size=512, bucket_m=25.0):
        self.arcs = np.asarray(network.arcs, dtype=np.int64)
        self.lengths = np.asarray(network.arc_len, dtype=float)
        self.a = np.asarray(network.arc_a, dtype=float)
        self.b = np.asarray(network.arc_b, dtype=float)
        n = len(network.node_xy)
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self.targets = self.arcs[order, 1]
        self.weights = self.lengths[order]
        self.starts = np.r_[0, np.cumsum(np.bincount(self.arcs[:, 0], minlength=n))]
        self.cache = OrderedDict()
        self.cache_size, self.bucket_m = cache_size, bucket_m

    def distances(self, source, cutoff):
        # Round up: cache reuse must never exclude a path inside the real cutoff.
        bucket = math.ceil(cutoff / self.bucket_m)
        key = (int(source), bucket)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        limit = bucket * self.bucket_m
        best = {int(source): 0.0}
        queue = [(0.0, int(source))]
        while queue:
            dist, node = heapq.heappop(queue)
            if dist != best[node]:
                continue
            for k in range(self.starts[node], self.starts[node + 1]):
                target, nd = int(self.targets[k]), dist + self.weights[k]
                if nd <= limit and nd < best.get(target, math.inf):
                    best[target] = nd
                    heapq.heappush(queue, (nd, target))
        self.cache[key] = best
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return best

    def offsets(self, ids, point):
        delta = self.b[ids] - self.a[ids]
        den = np.einsum('ij,ij->i', delta, delta)
        frac = np.divide(np.einsum('ij,ij->i', point - self.a[ids], delta), den,
                         out=np.zeros(len(ids)), where=den > 0)
        return np.clip(frac, 0, 1) * self.lengths[ids]
