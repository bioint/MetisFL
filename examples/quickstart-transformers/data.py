
import random
from typing import Tuple

import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch


MODEL_URL = "distilbert-base-uncased"

def load_data() -> Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader]:
    """Load imdb dataset and select 10 random samples for training and testing."""

    dataset = datasets.load_dataset("imdb")
    dataset = dataset.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)
    preprocess_data = lambda examples: tokenizer(examples["text"], truncation=True)

    encoded_dataset = dataset.map(preprocess_data, batched=True)
    encoded_dataset.set_format("torch")

    train_sample_indices = random.sample(range(len(dataset["train"])), 10)
    test_sample_indices = random.sample(range(len(dataset["train"])), 10)

    encoded_dataset["train"] = encoded_dataset["train"].select(train_sample_indices)
    encoded_dataset["test"] = encoded_dataset["test"].select(test_sample_indices)
    encoded_dataset = encoded_dataset.remove_columns("text")
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainloader = DataLoader(
        encoded_dataset["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=collator,
    )

    testloader = DataLoader(
        encoded_dataset["test"],
        batch_size=8,
        collate_fn=collator,
    )

    return trainloader, testloader