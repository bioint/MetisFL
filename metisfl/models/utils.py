
import numpy as np


def calc_mean_wall_clock(wall_clock):
    return np.mean(wall_clock) * 1000


def get_num_of_epochs(dataset_size: int, batch_size: int, total_steps: int) -> int:
    steps_per_epoch = np.ceil(np.divide(dataset_size, batch_size))
    epochs_num = 1
    if total_steps > steps_per_epoch:
        epochs_num = int(np.ceil(np.divide(total_steps, steps_per_epoch)))
    return epochs_num
