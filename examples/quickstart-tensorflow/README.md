# 🚀 MetisFL Quickstart: TensorFlow

<div align="center">
<picture>
  <img 
    style="border: 1px solid black; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.5);"
  alt="MetisFL TensforFlow Quickstart" src="https://docs.nevron.ai/img/gif/quickstart-tensorflow.gif">
</picture>
</div>

&nbsp;

This example shows how to use MetisFL to train a federated model. The example uses Tensorflow as the machine learning framework and the CIFAR10 dataset. This guide describes the main steps and the full scripts can be found in the [examples/quickstart-tensorflow](https://github.com/NevronAI/metisfl/tree/main/examples/quickstart-tensorflow) directory.

It is recommended to run this example in an isolated Python environment. You can create a new environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/).

The example consists of the Controller and 3 learners all running in the same machine. The Controller is responsible for sending training and evaluation tasks to the learners and for aggregating the model parameters. The learners are responsible for training the model on the local dataset and communicating with the server. The main entry point to the MetisFL framework is the Driver. The Driver is responsible for coordinating the communication between the clients and the server, for initializing the weights of the shared model, monitoring the federated training and shutting down the system when the training is done.

## ⚙️ Prerequisites

Before running this example, please make sure you have installed the MetisFL package

```bash
pip install metisfl
```

The default installation of MetisFL does not include any backend. Since this example uses Tensorflow, you need to install that if you haven't already.

```bash
pip install tensorflow
```

## 💾 Dataset

The dataset used in this example is the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The dataset is downloaded and prepared in the `load_data` function.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

To prepare the dataset for simulated federated learning, we need to split it into chunks and distribute them to the learners. We can use the `iid_partition` function in `metisfl.common.utils` module to do this.

```python
from metisfl.common.utils import iid_partition
x_chunks, y_chunks = iid_partition(x_train, y_train, num_partitions=3, seed=1990)
```

This function takes the dataset and splits it into `num_partitions` chunks. The optional `seed` parameter is used to control the randomness of the split and can be used to reproduce the same split. It produces independent and identically distributed (IID) chunks of the dataset. By keeping the seed constant, we can ensure that the same chunks are produced every time.

## 🧠 Model

The model used in this example is a simple Dense Neural Network with two hidden layers and a softmax output layer. The model is defined in the `get_model` function in the [model.py](https://github.com/NevronAI/metisfl/blob/main/examples/quickstart-tensorflow/model.py) file. The function allows for some flexibility in the model definition and can be used to define different models or tune this one.

```python
def get_model(
        input_shape: Tuple[int] = (32, 32, 3),
        dense_units_per_layer: List[int] = [256, 128],
        num_classes: int = 10) -> tf.keras.Model:

    Dense = tf.keras.layers.Dense
    Flatten = tf.keras.layers.Flatten
    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))

    for units in dense_units_per_layer:
        model.add(Dense(units=units, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model
```

Note that on the top of the file we set the following Tensorflow configuration:

```python
tf.config.experimental.set_memory_growth(gpu, True)
```

This is required to avoid Tensorflow allocating all the GPU memory to the first learner. For more information, please have a look at the [Tensorflow documentation](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

## 🎛️ MetisFL Controller

The main abstraction of the server is called MetisFL Controller. The Controller is responsible for sending training and evaluation tasks to the learners and for aggregating the model parameters. The entrypoint for the Controller is `Controller` class found [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/controller/controller_instance.py#L10).

```python
controller_params = ServerParams(
    hostname="localhost",
    port=50051,
)

controller_config = ControllerConfig(
    aggregation_rule="FedAvg",
    scheduler="Synchronous",
    scaling_factor="NumTrainingExamples",
)

model_store_config = ModelStoreConfig(
    model_store="InMemory",
    lineage_length=0
)

controller = Controller(
    server_params=controller_params,
    controller_config=controller_config,
    model_store_config=model_store_config,
)
```

The ServerParams define the hostname and port of the Controller and, optionally, the paths to the root certificate, server certificate and private key. The ControllerConfig defines the aggregation rule, scheduler and model scaling factor. For the full set of options in the ControllerConfig please have a look [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/common/types.py#L99). Note that the "NumTrainingExamples" scaling factor requires that the Learner instance provides the size of its training dataset at initialization. Finally, this example uses an "InMemory" model store with no eviction (`lineage_length=0`).

## 👨‍💻 MetisFL Learner

The MetisFL Learner is responsible for training the model on the local dataset and communicating with the server. For the training tasks, the communication between the Learner and Controller is asynchronous at the protocol level. The Controller sends the training task to the Learner and the Learner sends back a simple acknowledgement of receiving the task. When the Learner finishes the task it sends the results back to the Controller by calling its `TrainDone` endpoint. For the evaluation tasks, the communication is synchronous at the protocol level. The Controller sends the evaluation task to the Learner and waits for the results using the same channel. The abstract class that defines the Learner can be found [here](https://github.com/NevronAI/metisfl/blob/main/metisfl/learner/learner.py).

The main reason for choosing such an architecture is the fact that training task normally take long to complete and, by releasing the communication channel after sending the task, the Controller can scale and support thousands of Learners without having to maintain a connection with each one of them.

For this quickstart example, the Learner that we are using is the following:

```python
class TFLearner(Learner):

    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.model = get_model()
        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, parameters):
        self.model.set_weights(parameters)

    def train(self, parameters, config):
        self.model.set_weights(parameters)
        batch_size = config["batch_size"] if "batch_size" in config else 64
        epochs = config["epochs"] if "epochs" in config else 3
        res = self.model.fit(x=self.x_train, y=self.y_train,
                             batch_size=batch_size, epochs=epochs)
        parameters = self.model.get_weights()
        return parameters, res.history

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return {"accuracy": float(accuracy), "loss": float(loss)}
```

The `get_weights()` and `set_weights()` functions are used to get and set the model parameters. The `train()` and `evaluate()` functions are used to train and evaluate the model, respectively, on the local dataset. Both of these methods will get the model parameters and a configuration dictionary as input. The configuration dictionary is used to pass additional parameter, such as the batch size, local epochs, optimizer parameters, etc. It is up to the learner to decide how to use these parameters. Finally, note that the `train` method returns the `num_training_examples` as part of the training logs. This is required for the `NumTrainingExamples` scaling factor previously mentioned in the Controller section.

## 🚦 MetisFL Driver

The MetisFL Driver is the main entry point to the MetisFL framework. It is responsible for coordinating the communication between the clients and the server, for initializing the weights of the shared model, monitoring the federated training and shutting down the system when the training is done. The Driver is initialized with the Controller and the Learners.

```python
def get_learner_server_params(learner_index, max_learners=3):
    ports = list(range(50002, 50002 + max_learners))

    return ServerParams(
        hostname="localhost",
        port=ports[learner_index],
    )

termination_signals = TerminationSingals(federation_rounds=5)
learners = [get_learner_server_params(i) for i in range(max_learners)]

session = DriverSession(
    controller=controller_params,
    learners=learners,
    termination_signals=termination_signals,
)

logs = session.run()
```

The TerminationSignals control when the federated training is stopped. For this example, we will stop the training when we reach 5 federation rounds. For other possible termination signals, please have a look at the class definition and the docstring here [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/common/types.py#L18).

## 🎬 Running the example

To run the example, you need to open one terminal for the Controller, one terminal for each Learner and one terminal for the Driver. First, start the Controller.

```bash
python controller.py
```

Then, start the Learners.

```bash
python learner.py -l ID
```

where `ID` is the numerical id of the Learner (1,2,3). Please make sure to start the controller before the Learners otherwise the Learners will not be able to connect to the Controller. Finally, start the Driver.

```bash
python driver.py
```

The driver will run the federated training for 5 rounds and then stop. The training logs will be save in the `results.json` file in the current directory.

## 🚀 Next steps

Congratulations👏 You have successfully run your first federated learning experiment with MetisFL and you should see an output similar to the image at the top of this page. There are multiple way with which you can experiment and extend this example. Here are some ideas:

- Try varying the number of learners and see how you partition the dataset.
- Try different models or different hyperparameters on the existing model and see how the federated training converges.
- Try different aggregation rules and schedulers.

Please share your results with us or ask any questions that you might have on our [Slack channel](https://nevronai.slack.com/archives/C05E9HCG0DB). We would love to hear from you!
