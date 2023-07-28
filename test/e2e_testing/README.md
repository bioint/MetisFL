### Simple Testing
```
python examples/keras_fashionmnist/main.py --env test/e2e_testing/template_fedavg.yaml

python examples/keras_fashionmnist/main.py --env test/e2e_testing/template_fedprox.yaml --opt="FedProx"
```

### SSL Testing

```
# Case-1: End-to-End test integration test with SSL only for the Controller.
python examples/keras_fashionmnist/main.py --env test/e2e_testing/template_ssl_controller.yaml

# Case-2: End-to-End test integration test with SSL for both the Controller and the Learner.
python examples/keras_fashionmnist/main.py --env test/e2e_testing/template_ssl_controller_learner.yaml
```

### Homomorphic Library Testing

```
# Case-1: Need to pass all secure aggregation test cases at the controller.
bazelisk run //metisfl/controller/aggregation:secure_aggregation_test  

# Case-2: No failure must occur when running the following python CKKS demo.
python metisfl/encryption/pybind_ckks_demo.py

# Case-3: End-to-End test integration test (Secure Aggregation with CKKS + SSL).
python examples/keras_fashionmnist/main.py --env test/e2e_testing/template_ssl_secagg_ckks.yaml
```
