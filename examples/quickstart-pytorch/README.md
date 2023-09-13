# 🚀 MetisFL Quickstart: PyTorch

<div align="center">
<picture>
  <img 
    style="border: 1px solid black; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.5);"
  alt="MetisFL TensforFlow Quickstart" src="https://docs.nevron.ai/img/gif/quickstart-pytorch.gif">
</picture>
</div>

&nbsp;

This example shows how to use MetisFL to train a Pytorch model in a simulated federated learning setting using MetisFL. The guide describes the main steps and the full scripts can be found in the [examples/quickstart-pytorch](https://github.com/NevronAI/metisfl/tree/main/examples/quickstart-pytorch) directory.

It is recommended to run this example in an isolated Python environment. You can create a new environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/).

## ⚙️ Prerequisites

Before running this example, please make sure you have installed the MetisFL package

```bash
pip install metisfl
```

The default installation of MetisFL does not include any backend. This example uses Pytorch as a backend as well as torchvision to load the CIFAR10 dataset. Both can be installed using pip.

```bash
pip install torch torchvision
```

## 💾 Dataset

The dataset we use is the CIFAR10 and the example is based on the model training example in the [Pytorch documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). First, we load the dataset and split it into `num_learners` chunks.

```python
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
```

To split the dataset we user the `iid_partition` function from the `metisfl.common.utils` module. This function takes the dataset and splits it into `num_partitions` chunks. The optional `seed` parameter is used to control the randomness of the split and can be used to reproduce the same split. It produces independent and identically distributed (IID) chunks of the dataset. Note that the data are transformed channels first (NCHW) as expected by Pytorch.

## 🧠 Model

The model used in this example is a simple CNN and is defined in the `model.py` file.

```python
class Model(nn.Module):
    """A simple CNN for CIFAR-10."""

    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
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
    losses = []
    accs = []
    for _ in range(epochs):
        for images, labels in self.trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total

            losses.append(loss.item())
            accs.append(accuracy)

    metrics = {
        "accuracy": np.mean(accs),
        "loss": np.mean(losses),
    }
    metadata = {
        "num_training_examples": len(self.trainset),
    }
    return self.get_weights(), metrics, metadata
```

```python
def evaluate(self, parameters, config):
    self.set_weights(parameters)

    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in self.testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = self.model(images)
            loss += self.criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    loss = loss / total

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
    federation_rounds=5)
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

Congratulations 👏 you have successfully run your first MetisFL federated learning experiment using Pytorch! And you should see an output similar to the image on the top of this page. You may notice that the performance of the model is not that good. You can try to improve it by experimenting both the the federated learning parameters (e.g., the number of learners, federation rounds, aggregation rule) as well as with the typical machine learning parameters (e.g., learning rate, batch size, number of epochs, model architecture).

Please share your results with us or ask any questions that you might have on our [Slack channel](https://nevronai.slack.com/archives/C05E9HCG0DB). We would love to hear from you!
