
from collections import namedtuple


ModelWeightsDescriptor = \
    namedtuple('ModelWeightsDescriptor',
               ['weights_names', 'weights_trainable', 'weights_values'])


LearningTaskStats = namedtuple('LearningTaskStats', [
    "train_stats",
    "completed_epochs",
    "global_iteration",
    "validation_stats",
    "test_stats",
    "completes_batches",
    "batch_size",
    "processing_ms_per_epoch",
    "processing_ms_per_batch"
])
