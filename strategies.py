import enum
from abc import ABC, abstractmethod
from typing import Iterable

from torch.utils.data import Dataset
import random


class BatchWrapper(ABC):

    @abstractmethod
    def get_batch_iter(self, datasets: Iterable[Dataset], batch_size: int):
        pass


class SequentialBatching(BatchWrapper):
    def get_batch_iter(self, datasets: Iterable[Dataset], batch_size: int):
        for d in datasets:
            for x in d:
                yield x


class MixedBatching(BatchWrapper):
    def get_batch_iter(self, datasets: Iterable[Dataset], batch_size: int):
        iters = list(map(iter, datasets))
        while iters:
            it = random.choice(iters)
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)


class ProportionalBatching(BatchWrapper):
    def get_batch_iter(self, datasets: Iterable[Dataset], batch_size: int):
        iters = list(map(iter, datasets))
        di = 0
        while iters:
            it = iters[di]
            i = 0
            try:
                rng = round(batch_size / len(iters))
                while i < rng:
                    x = next(it)
                    i += 1
                    yield x
                else:
                    di = (di + 1) % len(iters)
            except StopIteration:
                iters.remove(it)
                if di >= len(iters):
                    di = 0


class TaskBasedBatching(BatchWrapper):
    def get_batch_iter(self, datasets: Iterable[Dataset], batch_size: int):
        iters = list(map(iter, datasets))
        di = 0
        while iters:
            it = iters[di]
            i = 0
            try:
                while i < batch_size:
                    x = next(it)
                    i += 1
                    yield x
                else:
                    di = (di + 1) % len(iters)
            except StopIteration:
                iters.remove(it)
                if di >= len(iters):
                    di = 0


class BatchingStrategy(enum.Enum):
    SEQUENTIAL = SequentialBatching()
    MIXED_RANDOM = MixedBatching()
    MIXED_PROPORTIONAL = ProportionalBatching()
    TASK_PER_BATCH = TaskBasedBatching()
