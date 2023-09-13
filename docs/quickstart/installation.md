
## Installation

It is recommended to install MetisFL in a virtual environment. To get started, first install ensure that you system meets the requirements:

- Python 3.8 to 3.10
- A x86_64 Linux distro (tested on Ubuntu Focal)

Install the `metisfl` Python package by running:

```shell
pip install metisfl
```

and then clone the repository on you local machine:

```shell
git clone https://github.com/NevronAI/metisfl.git
```

Navigate on the root project folder and run:

```shell
python examples/keras/fashionmnist.py
```

Congratulations! You are now running your first federated learning experiment using MetisFL!

## MetisFL Configuration

The federated environment is composed of a Controller, one or more Learners as well as configuration for the encryption, model store, termination signals etc. This section provides an overview those configuration options and how to use them.

### Controller

This specifies the configuration of the Controller/Aggregator. The following configuration options are currently supported:

```python
@dataclass
class ControllerConfig(object):
    """Configuration for the global training. Sets basic parameters reqired for the federated training.

    Parameters
    ----------
    aggregation_rule : str
        The aggregation rule to use. Must be one of the following: ["FedAvg", "FedRec", "FedStride", "SecAgg"].
    scheduler : str
        The scheduler to use. Must be one of the following: ["Synchronous", "Asynchronous", "SemiSynchronous"].
    scaling_factor : str
        The scaling factor to use. Must be one of the following: ["NumTrainingExamples", "NumCompletedBatches", "NumParticipants"].
    participation_ratio : Optional[float], (default=1.0)
        The participation ratio to use. Defaults to 1.0.
    stride_length : Optional[int], (default=None)
        The stride length to use. Required if the aggregation rule is FedStride.
    crypto_context : Optional[str], (default=None)
        The HE crypto context file to use. Required if the aggregation rule is SecAgg.
    semi_sync_lambda : Optional[float], (default=None)
        The semi-sync lambda to use. Required if the scheduler is SemiSynchronous.
    semi_sync_recompute_num_updates : Optional[bool], (default=None)
        Whether to recompute the number of updates. Required if the scheduler is SemiSynchronous.

    """

    aggregation_rule: str
    scheduler: str
    scaling_factor: str
    participation_ratio: Optional[float] = 1.0
    stride_length: Optional[int] = None
    crypto_context: Optional[str] = None
    batch_size: Optional[int] = None
    scaling_factor_bits: Optional[int] = None
    semi_sync_lambda: Optional[float] = None
    semi_sync_recompute_num_updates: Optional[bool] = None
```

The `aggretation_rule`, `scheduler` and `scaling_factor` are required parameters. Some important constraint to note are the following:

- If the `scaling_factor` is `NumCompletedBatches` then the `num_completed_batches` must be returned by the Learner's `train` method.
- If the `scaling_factor` is `NumTrainingExamples` then the `num_training_examples` must be returned by the Learner's `train` method.

Note that the `batch_size` mentioned here refers to the CKKS encryption scheme and not the batch size used for training the model.

### Encryption Config

This section specifies the configuration of the encryption scheme used for the secure aggregation. This part is only required if the aggregation rule is `SecAgg`.

```python
@dataclass
class EncryptionConfig(object):
    """Configures the encryption scheme on the learner side.

    Parameters
    ----------
    he_scheme : str
        The HE scheme to use. Must be one of the following: ["CKKS"].
    batch_size : Optional[int], (default=None)
        The batch size to use. Required if the HE scheme is CKKS.
    scaling_factor_bits : Optional[int], (default=None)
        The scaling factor bits to use. Required if the HE scheme is CKKS.
    crypto_context : Optional[str], (default=None)
        The HE crypto context file to use. Required if the HE scheme is CKKS.
    public_key : Optional[str], (default=None)
        The HE public key file to use. Required if the HE scheme is CKKS.
    private_key : Optional[str], (default=None)
        The HE private key file to use. Required if the HE scheme is CKKS.

    """
    he_scheme: Optional[str] = "CKKS"
    batch_size: Optional[int]
    scaling_factor_bits: Optional[int]
    crypto_context: Optional[str]
    public_key: Optional[str]
    private_key: Optional[str]
```

Currently, only the CKKS scheme is supported. All parameters are required and must match the parameters used by the Controller. If not, the Learners will fail to properly encrypt/decrypt the model weights and and Controller will fail to properly aggregate the encrypted model weights.

### Termination Signals

The `TerminationSignals` section of the configuration file specifies the conditions under which the federated learning process will terminate. The 3 main termination signals are the number of federation rounds, the execution time and the evaluation metric.

```python
@dataclass
class TerminationSingals(object):
    """Set of termination signals for the federated training. Controls when the training is terminated.

    Parameters
    ----------
    federation_rounds : Optional[int], (default=None)
        Number of federation rounds to run. I
    execution_cutoff_time_mins : Optional[int], (default=None)
        Maximum execution time in minutes.
    evaluation_metric : Optional[str], (default=None)
        The evaluation metric to use for early stopping.
        The metric must be returned by the Learner's `evaluate` method.
    evaluation_metric_cutoff_score : Optional[float], (default=None)
        The evaluation metric cutoff score for early stopping.

    """

    federation_rounds:  Optional[int] = None
    execution_cutoff_time_mins: Optional[int] = None
    evaluation_metric: Optional[str] = None
    evaluation_metric_cutoff_score: Optional[float] = None
```

At least one of the above must be defined, otherwise the training will not terminate. Two important constraints to note are the following:

- The `federation_rounds` is not available when using the `Asynchronous` scheduler, because the number of federation rounds is not defined in this case.
- The `evaluation_metric` must be returned by the Learner's `train` method and `evaluate` method.

### Model Store

This section defines where the controller stores the model weights. Currently, the following options are supported:

```python
@dataclass
class ModelStoreConfig(object):
    """Configuration for the model store. Controls how the model store is used.

    Parameters
    ----------
    model_store : Optional[str], (default="InMemory")
        The model store to use. Must be one of the following: ["InMemory", "Redis"].
    lineage_length : Optional[int], (default=0)
        The max number of models to store before start evicting models. If 0, no eviction is performed.
    model_store_hostname : Optional[str], (default=None)
        The hostname of the model store. Required if the model store is Redis.
    model_store_port : Optional[int], (default=None)
        The port of the model store. Required if the model store is Redis.


    Raises
    ------
    ValueError
        Value error is raised in the following cases:
        - If the model store is not one of the following: ["InMemory", "Redis"].
        - If the model store is Redis and the hostname or port are not specified.
    """

    model_store: Optional[str] = "InMemory"
    lineage_length: Optional[int] = 0
    model_store_hostname: Optional[str] = None
    model_store_port: Optional[int] = None
```

The `model_store` can be either `InMemory` or `Redis`. If `Redis` is selected, then the `model_store_hostname` and `model_store_port` must be specified. Note that the Redis model store is currently in beta.
