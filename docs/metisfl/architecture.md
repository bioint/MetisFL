
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-Components-Internal-02.webp" width="700px">
    <img alt="MetisFL Components Internal" src="https://docs.nevron.ai/img/dark/MetisFL-Components-Internal-01.webp" width="700px">
    </picture>
</div>

## Federation Controller

The Federation Controller is responsible for selecting and scheduling training and evaluating tasks to the federation learners, for receiving, aggregating and storing the model updates and for storing the training logs and metadata. For the training tasks, the communication between the Learner and Controller is asynchronous at the protocol level. The Controller sends the training task to the Learner and the Learner sends back a simple acknowledgement of receiving the task. When the Learner finishes the task it sends the results back to the Controller by calling its TrainDone endpoint. For the evaluation tasks, the communication is synchronous at the protocol level, i.e., the Controller sends the evaluation task to the Learner and waits for the results using the same channel.

- Aggregation Rules: The Controller is responsible for aggregating the local models of the learners. The aggregation rules are defined by the system user and can be either a simple average of the local models or a more complex aggregation rule, e.g., weighted average, FedAvg, FedProx, etc. Currently, we support the following aggregation rules:

  - Federated Average (FedAvg)
  - Federated Recency (FedRec)
  - Federated Stride (FedStride)
  - Secure Aggregation (SecAgg)

- Learner Schedulers: A scheduler defines the subset of the learner which will receive the next task. Currently, we support the following schedulers:

  - Synchronous Scheduler: All learners receive the next task at the same time.
  - Asynchronous Scheduler: Each learner receives the next task when it finishes the previous one.
  - SemiSynchronous Scheduler: Same as the Synchronous Scheduler, but the each learners runtime performance is taken into account when scheduling the next task.

- Model Store: The Model Store is responsible for storing the model updates received from the learners. The Model Store can be either a local or a remote storage. Currently, we support the following storage options:
  - In-Memory: The model updates are stored in the memory of the Controller.
  - Redis (experimental): The model updates are stored in a Redis database.

## Federation Learner

The main abstraction of the client is called MetisFL Learner. The MetisFL Learner is responsible for training the model on the local dataset and communicating with the server. The abstract class that defines the Learner can be found [here](https://github.com/NevronAI/metisfl/blob/main/metisfl/learner/learner.py). The abstract class is defined as follows:

```python
class Learner(ABC):
    """Abstract class for all MetisFL Learners. All Learners should inherit from this class."""

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Returns the weights of the model as a list of numpy arrays.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays representing the weights of the model.
        """
        return np.array([])

    @abstractmethod
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Sets the weights of the model using the given weights.

        Parameters
        ----------
        weights : List[np.ndarray]
            A list of numpy arrays representing the weights of the model to be set.

        """
        return False

    @abstractmethod
    def train(
        self,
        weights: List[np.ndarray],
        params: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Trains the model using the given training parameters.

        Parameters
        ----------
        weights : List[np.ndarray]
            A list of numpy arrays representing the weights of the model to be trained.
        params : Dict[str, Any]
            A dictionary of training parameters.

        Returns
        -------
        Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]
            A tuple containing the following:
                - A list of numpy arrays representing the weights of the model after training.
                - A dictionary of the metrics computed during training.
                - A dictionary of training metadata, such as the number
                    of completed epochs, batches, processing time, etc.
        """
        return [], {}

    @abstractmethod
    def evaluate(
        self,
        model: List[np.ndarray],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluates the given model using the given evaluation parameters."""
        return {}
```

Each MetisFL learner must implement the following methods:

- `set_weights`: This method is called by the Controller to set the initial model weights. The Learner must implement the logic to set the weights of the local model.
- `get_weights`: This method is called by the Controller to get the current model weights. The Learner must implement the logic to return the weights of the local model.
- `train`: This method is called by the Controller to initiate the training task. The input to this method is the current model weights and a dictionary of hyperparameters. The Learner must implement the training logic and return the updated model weights to the Controller.
- `evaluate`: This method is called by the Controller to initiate the evaluation task. The Learner must implement the evaluation logic and return the evaluation metrics to the Controller.

At its core, the Learner implements both a gRPC server and a client. The former one is used to receive the training and evaluation tasks from the Controller and the latter one is used to send the results back to the Controller. Additionally, a task manager, using separate python sub-processes, allows us to run multiple, isolated training and evaluation tasks in parallel.

## Federation Driver

The Federation Driver is responsible for orchestrating the execution of the federated learning workflow. It has 3 main components:

- Driver Session: The entry-point to the Driver app. Acts as a container for the Monitor and gRPC clients mentioned below.

* Federation Monitor: It is responsible for periodically pinging the controller to request the training logs and metadata. The logs and metadata are used, alongside the termination signal defined by the user, to determine when the federated learning workflow should be terminated.

* gRPC Clients: The Driver implements two types gRPC clients: one to connect to the Controller and one to connect to each Learner. At initialization, the Driver requests the initial model weights from a randomly selected Learner and distributes them to the other participants. This is to ensure that all Learners start with the same initial model weights.
