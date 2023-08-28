
from collections import namedtuple


LearningTaskStats = namedtuple('LearningTaskStats', [
    "global_iteration",
    "train_stats",
    "validation_stats",
    "test_stats",
    "completed_epochs",
    "completes_batches",
    "batch_size",
    "processing_ms_per_epoch",
    "processing_ms_per_batch"
])
