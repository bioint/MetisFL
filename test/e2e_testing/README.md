## General Guidelines
- The e2e testing is conducted using a simple MLP model over the FashionMNIST dataset.

- By default each e2e testing template environment consists of 1 controller and 1 learner; except if stated otherwise, e.g., *_3learners.yaml.

- For all synchronous and semi-synchronous experiments, the termination signal is the number of federation rounds; currently set to 3.

- For asynchronous experiments, the termination signal is accuracy or execution time.



### Simple Testing
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_fedavg.yaml

python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_fedavg_3learners.yaml --num_learners=3

python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_fedprox.yaml --opt="FedProx"
```

### Model Store Testing

```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/modelstore/template_inmemory_store_all_models_3learners.yaml --num_learners=3

python examples/keras_fashionmnist/main.py --env test/e2e_testing/modelstore/template_inmemory_store_all_encrypted_models_3learners.yaml --num_learners=3
```

> For the Redis experiments, we need first initialize the Redis instance by running:
`service redis-server start` inside the container/server MetisFL is running. Then whenever we need to run a 
new experiment to test the system behavior we need clear the database before executing that experiment. To do so,
we start the cli client (`redis-cli`) and then we run the `FLUSHALL` command.
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/modelstore/template_redis_store_all_models_3learners.yaml --num_learners=3

python examples/keras_fashionmnist/main.py --env test/e2e_testing/modelstore/template_redis_store_all_encrypted_models_3learners.yaml --num_learners=3
```

### Asynchronous Testing
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/asynchronous/template_accuracy_cutoff_3learners.yaml

python examples/keras_fashionmnist/main.py --env test/e2e_testing/asynchronous/template_time_cutoff_3learners.yaml --opt="FedProx" --num_learners=3
```

### SemiSynchronous Testing
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/semisynchronous/template_fedavg_3learners.yaml --num_learners=3

```

### SSL Testing

```
# Case-1: End-to-End test integration test with SSL only for the Controller.
python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_ssl_controller.yaml

# Case-2: End-to-End test integration test with SSL for both the Controller and the Learner.
python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_ssl_controller_learner.yaml
```

### Homomorphic Library Testing

```
# Case-1: Need to pass all secure aggregation test cases at the controller.
bazelisk run //metisfl/controller/aggregation:secure_aggregation_test  

# Case-2: No failure must occur when running the following python CKKS demo.
python metisfl/encryption/pybind_ckks_demo.py

# Case-3: End-to-End test integration test (Secure Aggregation with CKKS + SSL).
python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_ssl_secagg_ckks.yaml

# Case-4: End-to-End test integration test (Secure Aggregation with CKKS + SSL) for 3 learners.
python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_ssl_secagg_ckks_3learners.yaml --num_learners=3
```
