import jax.numpy as jnp
from collections import defaultdict
import time


class LastItem:
    def __init__(self):
        self.val = 114514.1919810

    def append(self, v):
        self.val = v

    def get(self):
        return self.val


class Avger(list):
    def get(self):
        return sum(self) / len(self) if len(self) > 0 else 114514.1919810


class MyMetrics:
    def __init__(self, reduction="last"):
        self.reduction_cls = {
            "last": LastItem,
            "avg": Avger,
        }[reduction]
        self.metrics = defaultdict(self.reduction_cls)

    def update(self, metrics):
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def compute(self, *keys):
        if len(keys) == 0:
            return {k: float(jnp.mean(v.get())) for k, v in self.metrics.items()}
        else:
            raise NotImplementedError
            # return tuple(self.metrics[k].get() for k in keys)

    def reset(self):
        self.metrics = defaultdict(self.reduction_cls)

    def compute_and_reset(self, *keys):
        a = self.compute(*keys)
        self.reset()
        return a


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f"{self.elapse_with_reset():.2f} s"
