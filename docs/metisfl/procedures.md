
## Federation Lifecycle
Here, we present the internal mechanisms of MetisFL with respect to federation initialization, and federated model training and evaluation using a synchronous communication protocol. The Federation lifecycle consists of three core steps: _initialization, monitoring, shutdown_:

<p align="center">
<picture>
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-Procedures-FederationLifecycle-02.webp" width="700px">
    <img alt="Federation Lifecycle" src="https://docs.nevron.ai/img/dark/MetisFL-Procedures-FederationLifecycle-01.webp" width="700px">
</picture>
</p>

We first initialize the federation by creating the controller and the learners. The Controllers/Learners can either be in a single machine for simulation purposes or in different machines for a distributed setup. Upon startup, the learners then register with the controller (join federation) and await for training tasks. It is important to first start the controller and then the learners, otherwise the learners will not be able to register with the controller.

Then, we initialize the driver process that will orchestrate the federation workflow. The driver initializes the federation by pinging a randomly selected learner for the initial weights and distributing the model to all learners. After the model initialization, the driver starts the federated training and evaluation. At every federation round the global model is sent for training and subsequently for evaluation to the participating learners. At this point, the driver monitors the lifecycle of the federation and periodically pings remote processes for their status. If any of the federated training termination criteria are met, such as the execution wall-clock time or number of federation rounds, then the driver sends a shutdown signal to all processes (learners first, controller second).

## Training Round

Before the training round starts, the controller creates/defines the model training task and selects the learners who will participate in model training. Once the learners have been selected the train task scheduler sends the training task to every participating learner.

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-Procedures-TrainingRound-02.webp" width="700px">
        <img alt="Training Round with Requests" src="https://docs.nevron.ai/img/dark/MetisFL-Procedures-TrainingRound-01.webp" width="700px">
    </picture>
</p>

The Learner receives the task through the Learner Servicer process and submits the training task to the training task pool executor running in the background. Upon task submission, the executor replies an Acknowledgement message that the servicer relays back to the controller. The status of the acknowledgement is **false** when the training task is not submitted, received or any unexpected failure occurs.

<p align="center">
<picture>
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-Procedures-TrainingRound-02.webp" width="700px">
    <img alt="Training Round with Requests" src="https://docs.nevron.ai/img/dark/MetisFL-Procedures-TrainingRound-01.webp" width="700px">
</picture>
</p>

The submitted training task is registered with a callback function that will handle the completed training task when it is completed When this occurs, the servicer sends a TrainDone request to the controller containing the learner's local model and any other execution metadata related to model training. Finally, the controller stores and aggregates all received local models. The training tasks are **asynchronous calls**, meaning that the controller does not wait for the training task to complete. The controller simply submits the task but the learner needs to inform the controller when its local training is complete.

## Evaluation Round

<div align="center">
<picture>
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-Procedures-EvaluationRound-02.webp" width="700px">
    <img alt="Evaluation Round with Requests" src="https://docs.nevron.ai/img/dark/MetisFL-Procedures-EvaluationRound-01.webp" width="700px">
</pictyre>
</div>

Similar to the training round, the evaluation round starts with the controller constructing the evaluation task and selecting the learners that will participate in the evaluation of the global model. Once these steps are defined, the evaluation task scheduler sends an EvaluateModel request to all participating learners and receives the respective model evaluations. The evaluation tasks are **synchronous calls**, meaning that the controller keeps the connection alive till the evaluation of the model is complete.

