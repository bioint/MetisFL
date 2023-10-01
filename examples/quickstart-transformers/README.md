# 🚀 MetisFL Quickstart: Transformers with Hugging Face

This example shows how to use MetisFL to fine-tune a DistilBERT model in a simulated federated learning setting using MetisFL. The guide describes the main steps and the full scripts can be found in the [examples/quickstart-transformers](https://github.com/NevronAI/metisfl/tree/main/examples/quickstart-transformers) directory. The example utilizes the transformers library from Hugging Face.

It is recommended to run this example in an isolated Python environment. You can create a new environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/).

## ⚙️ Prerequisites

Before running this example, please make sure you have installed the MetisFL package

```bash
pip install metisfl
```

The default installation of MetisFL does not include any backend. This example uses Pytorch as a backend. The transformers package among other associated packages from Hugging Face are required in order to load the IMDB dataset, the tokenizer and the pre-trained DistilBERT model. All of them can be installed using pip.

```bash
pip install datasets
pip install evaluate
pip install torch
pip install "transformers[torch]"
```

## 💾 Dataset

The dataset we use is the IMDB and the example is based on the model training example in the [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tasks/sequence_classification). First, we load the dataset and select 10 random samples for the test and test set.

```python
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
```

## 🧠 Model

In this example, we use is a pre-trained DistilBERT model. Along with the model, its tokenizer is needed to be fetched too. 

```python
MODEL_URL = "distilbert-base-uncased"

def get_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_URL,
        num_labels=2,
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)

    return model, tokenizer
```

## 👨‍💻 MetisFL Learner

The main abstraction of the client is called MetisFL Learner. The MetisFL Learner is responsible for training the model on the local dataset and communicating with the server. Following the [class](https://github.com/NevronAI/metisfl/blob/main/metisfl/learner/learner.py) that must be implemented by the learner, we first start by the `get_weights` and `set_weights` methods. These methods are used by the Controller to get and set the model parameters. The `get_weights` method returns a list of numpy arrays and the `set_weights` method takes a list of numpy arrays as input.

```python
def get_weights(self):
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

def set_weights(self, parameters):
    params = zip(self.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params})
    self.model.load_state_dict(state_dict, strict=True)
```

Then, we implement the `train` and `evaluate` methods. Both of them take the model weights and a dictionary of configuration parameters as input. The `train` method returns the updated model weights, a dictionary of metrics and a dictionary of metadata. The `evaluate` method returns a dictionary of metrics.

```python
def train(self, parameters, config):
    self.set_weights(parameters)
    epochs = config["epochs"] if "epochs" in config else 1
    lr = config["learning_rate"] if "learning_rate" in config else 2e-5
    weight_decay = config["weight_decay"] if "weight_decay" in config else 0.01

    optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    self.model.train()
    losses = []
    accs = []
    for epoch in range(epochs):
        for batch in self.trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predicted = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total

            losses.append(loss.item())
            accs.append(accuracy)
        print(f"Epoch {epoch+1}: Loss {np.mean(losses)} Accuracy {np.mean(accs)}")

    metrics = {
        "accuracy": np.mean(accs),
        "loss": np.mean(losses),
    }
    metadata = {
        "num_training_examples": len(self.trainloader.dataset),
    }
    return self.get_weights(), metrics, metadata
```

```python
def evaluate(self, parameters, config):
    self.set_weights(parameters)

    metric = evaluate.load("accuracy")
    loss = 0
    self.model.eval()
    for batch in self.testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        metric.add_batch(predictions=predictions, references=labels)
    loss /= len(self.testloader.dataset)
    accuracy = metric.compute()["accuracy"]

    print(f'Evaluation accuracy: {accuracy} and loss: {loss}')
    return {"accuracy": float(accuracy), "loss": float(loss)}
```

## 🎛️ MetisFL Controller

The Controller is responsible for send training and evaluation tasks to the learners and for aggregating the model parameters. The entrypoint for the Controller is `Controller` class found [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/controller/controller_instance.py#L10). The `Controller` class is initialized with the parameters of the Learners and the global training configuration.

```python
controller_params = ServerParams(
    hostname="localhost",
    port=50051,
)

controller_config = ControllerConfig(
    aggregation_rule="FedAvg",
    scheduler="Synchronous",
    scaling_factor="NumParticipants",
)

model_store_config = ModelStoreConfig(
    model_store="InMemory",
    lineage_length=0
)
```

The ServerParams define the hostname and port of the Controller and the paths to the root certificate, server certificate and private key. Certificates are optional and if not given then SSL is not active. The ControllerConfig defines the aggregation rule, scheduler and model scaling factor.

For the full set of options in the ControllerConfig please have a look [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/common/types.py#L99). Finally, this example uses an "InMemory" model store with no eviction (`lineage_length=0`). A positive value for `lineage_length` means that the Controller will start dropping models from the model store after the given number of models, starting from the oldest.

## 🚦 MetisFL Driver

The MetisFL Driver is the main entry point to the MetisFL application. It will initialize the model weights by requesting the model weights from a random learner and then distributing the weights to all learners and the controller. Additionally, it monitor the federation and will stop the training process when the termination condition is met.

```python
# Setup the environment.
termination_signals = TerminationSingals(
    federation_rounds=3)
learners = [get_learner_server_params(i) for i in range(max_learners)]
is_async = controller_config.scheduler == 'Asynchronous'

# Start the driver session.
session = DriverSession(
    controller=controller_params,
    learners=learners,
    termination_signals=termination_signals,
    is_async=is_async,
)

# Run
logs = session.run()
```

To see and experiment with the different termination conditions, please have a look at the TerminationsSignals class [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/common/types.py#L18).

## 🎬 Running the example

To run the example, you need to open one terminal for the Controller, one terminal for each Learner and one terminal for the Driver. First, start the Controller.

```bash
python controller.py
```

Then, start the Learners.

```bash
python learner.py -l X
```

where `X` is the numerical id of the Learner (1,2,3). Note that both the learner and driver scripts have been configured to use 3 learners by default. If you want to experiment with a different number of learners, you need to change the `max_learners` variable in both scripts. Also, please make sure to start the controller before the Learners otherwise the Learners will not be able to connect to the Controller.

Finally, start the Driver.

```bash
python driver.py
```

The Driver will start the training process and each terminal will show the progress. The experiment will run for 5 federation rounds and then stop. The logs will be saved in the `results.json` file in the current directory.

## 🚀 Next steps

Congratulations 👏 you have successfully run your first MetisFL federated learning experiment using DistilBERT and Hugging Face! You may notice that the performance of the model is not that good. You can try to improve it by experimenting both the the federated learning parameters (e.g., the number of learners, federation rounds, aggregation rule) as well as with the typical machine learning parameters (e.g., learning rate, batch size, number of epochs, model architecture).

Please share your results with us or ask any questions that you might have on our [Slack channel](https://nevronai.slack.com/archives/C05E9HCG0DB). We would love to hear from you!