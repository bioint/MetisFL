
from collections import namedtuple


ModelWeightsDescriptor = \
    namedtuple('ModelWeightsDescriptor',
               ['weights_names', 'weights_trainable', 'weights_values'])


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
