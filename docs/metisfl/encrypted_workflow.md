<div align="center">
 <picture> 
    <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-ExecutionFlow-WithEncryption-02.webp" width="700px">
    <img alt="MetisFL Execution Flow with FHE" src="https://docs.nevron.ai/img/dark/MetisFL-ExecutionFlow-WithEncryption-01.webp" width="700px">
</picture>
</div>

In a (centralized) federated learning environment, each learner trains on its local dataset for an assigned number of local epochs, and upon completion of its local training task, it sends the local model to the federation controller to compute the new global model.

In an encrypted (centralized) federation environment using Homomorphic Encryption (HE), the procedure is similar with the addition of three pivotal key steps: _encryption_, _encrypted-aggregation_, _decryption_.

During the **encryption** step, every learner encrypts its locally trained model with a Homomorphic Encryption (HE) scheme using the public key, and sends the encrypted model (ciphertext) to the controller. For each learner, its encrypted model is treated as a vector of ciphertext objects, each object corresponding to a model tensor. With this approach, the encrypted data is represented as a (concatenated) collection of flattened data-vectors, each of them representing the local data of a particular learner.

The controller receives all the encrypted local models, and then performs the **encrypted weighted-aggregation** to compute the new encrypted community model without ever decrypting any of the individual models.

Subsequently, the controller sends the new community model to all learners, and the learners **decrypt** it using the private key. Once the decryption is complete, the learners train the (new) decrypted model on their local data set and the entire procedure repeats.

MetisFL provides different private weighted aggregation functions built on top of the PALISADE/OpenFHE libraries. The default private weighted aggregation scheme is **CKKS**.
