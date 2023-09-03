# MetisFl Quikstart: Tensorflow

This example shows how to use MetisFL to train a Tensorflow in a simulated federated learning setting using MetisFL. This guide describes the main steps and the full scripts can be found in the [examples/quickstart-tensorflow](https://github.com/NevronAI/metisfl/tree/main/examples/quickstart-tensorflow) directory. 

## Prerequisites

Before running this example, please make sure you have installed the MetisFL package 

```bash
pip install metisfl
```

The default installation of MetisFL does not include any backend. Since this example uses Tensorflow, you need to install that if you haven't already. 

```bash
pip install tensorflow
```

## Dataset

The dataset used in this example is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The dataset can be easily downloaded using Keras. 

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

To prepare the dataset for simulated federated learning, we need to split it into chunks and distribute them to clients. We can use the `iid_partition` function in `metisfl.common.utils`  module to do this. 

```python
from metisfl.common.utils import iid_partition
x_chunks, y_chunks = iid_partition(x_train, y_train, num_partitions=3, seed=1990)
```

This function takes the dataset and splits it into `num_partitions` chunks. The optional `seed` parameter is used to control the randomness of the split and can be used to reproduce the same split. It produces independent and identically distributed (IID) chunks of the dataset.

## Model 

The model used in this example is a simple Dense Neural Network with two hidden layers and a softmax output layer. The model is defined in the `get_model` function in the [model.py](https://github.com/NevronAI/metisfl/blob/main/examples/quickstart-tensorflow/model.py) file. The function allows for some flexibilty in the model definition and can be used to define different models or tune this one.

## MetisFL Learner

The main abstraction of the client is called MetisFL Learner. 

## MetisFL Controller

The main abstraction of the server is called MetisFL Controller.

## MetisFL Driver

The MetisFL Driver is the main entry point to the MetisFL framework. It is responsible for coordinating the communication between the clients and the server. 

## Running the example

To run the example, you need to open one terminal for the Controller, one terminal for each Learner and one terminal for the Driver. First, start the Controller. 

```bash
python controller.py
```

Then, start the Learners. 

```bash
python learner.py --learner X --max-learners Y
```

where `X` is the numerical id of the Learner (0,1,2..) and `Y` is the total number of Learners (3 in this example). You can start as many Learners as you want. 

Finally, start the Driver. 

```bash
python driver.py
```

The Driver will start the training process and each terminal will show the progress of the training.