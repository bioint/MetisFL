Programming Model
=============================

MetisFL follows the successful programming model of Apache Spark, with the Federation Controller operating as the cluster manager of the federation, Federation Learners as the computing nodes, and the Federation Driver as the entry point of the federation launching various operations in parallel. The communication across internal and external services is established through appropriate RPC calls, using gRPC and protocol buffers.


<div align="center">
 <img 
    src="../img/MetisFL-Components-Internal.png" width="1000px", alt="MetisFL Components Overview">
</div>

</br>


* **Federation Driver:** parses the federated learning workflow defined by the system user and creates the Metis Context. The Metis Context is responsible for initializing and monitoring the federation cluster, initializing the original federation model state, defining the data loading recipe for each learner, and generating the security keys where needed, e.g., SSL certificates, FHE key pair. 

* **Federation Controller:** is responsible for selecting (Clients Selector) and delegating training and evaluating tasks (Task Schedulers) to the Federation Learners, and storing (Model Store) and aggregating (Model Aggregator) the learners' local models with or wihout encryption. The controller receives all incoming local models through rpc calls to the repsective endpoints of the gRPC server. Finally, since all these operations can be computationally intensive in the presence of large number of learners and/or very large models, the Federation Controller is implemented in C++17 in order to maximize the overall end-to-end performance. To enable easier development and enhance extensibility the Controller is also shipped with Python Bindings that wrap most of its core functionality.

* **Federation Learner(s):** are the cluster nodes participating in the training of the federated model. Each Federation Learner hosts two independent services, one for training (Model Trainer) and one for evaluating (Model Evaluator) the global model on its local private dataset. Both services live behind a gRPC server that is listening for incoming training and evaluation requests.



