By default each template file consists of 1 controller and 1 learner, except if stated otherwise, e.g., *_3learners.yaml
In all synchronous and semi-synchronous environments, the termination signal is the number of federation rounds, which is set to 3.
For asynchronous environments, we test the termination of the environment based on accuracy or execution time.
For all model store environments, we use a synchronous execution protocol over 3 learners and for 10 federation rounds.

### Simple Testing
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_fedavg.yaml

python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_fedavg_3learners.yaml --num_learners=3

python examples/keras_fashionmnist/main.py --env test/e2e_testing/synchronous/template_fedprox.yaml --opt="FedProx"
```

### Model Store Testing
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/modelstore/template_inmemory_store_all_models.yaml --num_learners=3

python examples/keras_fashionmnist/main.py --env test/e2e_testing/modelstore/template_inmemory_store_all_encrypted_models.yaml --num_learners=3
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
