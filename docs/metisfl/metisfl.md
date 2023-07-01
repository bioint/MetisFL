MetisFL
=============================
MetisFL is an easy-to-use scalable federated learning framework that follows the architectural principles of modularity, extensibility, and configurability. 

* **Modularity** refers to the development of functionally independent services (micro-services) that allow finer control of system components' interoperability. 

* **Extensibility** refers to the functional interface expansion of each service. 
  
* **Configurability** refers to the ease of deployment of new federated models and procedures. At its core, MetisFL uses Tensorflow and PyTorch as the backend ML/DL engines, and can be easily extended to support other backends as well such as JAX, MXNet.

<div align="center">
 <img name="components_overview" 
 src="../img/MetisFL-Components-Overview.png" width="1000px", alt="Components Overview.">
</div>

## Origins
MetisFL sprung up from the Information and Science Institute (ISI) in the University of Southern California (USC). It is backed by several years of Ph.D. and Post-Doctoral research and several publications in top-tier machine learning and system conferences.

## Application Domains
MetisFL is a general purpose, domain-agnostic federated learning system. It can be easily extended to support any type of federated training topology (centralized, peer-to-peer), protocol (synchronous, asynchronous) and federated algorithmic optimization. The framework has been extensively tested in challenging real-world and simulated federated (distributed) environments to train and learn federated models across various domains, such as in in Computer Vision, Natural Language Processing[[mathew2022](#mathew2022)], Neuroimaging[[stripelis2021](#stripelis2021)] and many other domains.


### References
<a name="mathew2022">[mathew2022]</a> Mathew, Joel, Dimitris Stripelis, and José Luis Ambite. "Federated Named Entity Recognition." arXiv preprint arXiv:2203.15101 (2022).

<a name="stripelis2021">[stripelis2021]</a> Stripelis, Dimitris, José Luis Ambite, Pradeep Lam, and Paul Thompson. "Scaling neuroscience research using federated learning." In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pp. 1191-1195. IEEE, 2021.
