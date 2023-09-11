
from typing import Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10

from metisfl.common.utils import iid_partition


def load_data(num_learners: int) -> Tuple:
    """Load CIFAR-10  and partition it into num_learners clients, iid."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    x_chunks, y_chunks = iid_partition(
        x_train=trainset.data, y_train=trainset.targets, num_partitions=num_learners)

    # Convert the numpy arrays to torch tensors and make it channels first
    x_chunks = [torch.Tensor(x).permute(0, 3, 1, 2) for x in x_chunks]
    y_chunks = [torch.Tensor(y).long() for y in y_chunks]
    trainset_chunks = [TensorDataset(x, y) for x, y in zip(x_chunks, y_chunks)]

    # Same for the test set
    test_data = torch.Tensor(testset.data).permute(0, 3, 1, 2)
    test_labels = torch.Tensor(testset.targets).long()
    testset = TensorDataset(test_data, test_labels)

    return trainset_chunks, testset
