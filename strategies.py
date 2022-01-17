import enum


class BatchingStrategy(enum.Enum):
    SEQUENTIAL = 1
    MIXED_RANDOM = 2
    MIXED_PROPORTIONAL = 3
    TASK_PER_BATCH = 4
