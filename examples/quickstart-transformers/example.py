import random
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from transformers import DataCollatorWithPadding
import evaluate

EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
LEARNING_RATE = 2e-5
MODEL_URL = "bert-base-uncased"
OUTPUT_DIR = "trained_model"

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_predictions: EvalPrediction):
    predictions, labels = eval_predictions
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

dataset = load_dataset("imdb")
dataset = dataset.shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)
preprocess_data = lambda examples: tokenizer(examples["text"], truncation=True)

encoded_dataset = dataset.map(preprocess_data, batched=True)
encoded_dataset.set_format("torch")

sample_indices = random.sample(range(len(dataset["train"])), 10)
encoded_dataset["train"] = encoded_dataset["train"].select(sample_indices)
encoded_dataset["test"] = encoded_dataset["test"].select(sample_indices)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_URL,
    num_labels=2,
).to(DEVICE)

args = TrainingArguments(
    OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

evaluation_results = trainer.evaluate()
print(evaluation_results)