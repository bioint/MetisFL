ExecutionProtocol_AND_Signals = [ 
    ("Synchronous", "FederationRounds", "")
    ("Synchronous", "ExecutionCutoffTimeMins", "")
    ("Synchronous", "EvaluationMetric", "")
    ("Asynchronous", "EvaluationMetric", "")
] 
ModelStoreConfig = [
    ("InMemory", "NoEviction")
    ("InMemory", "NoEviction")
    ]

EncryptionScheme = [
    ("CKKS", 4096, 52),
]

GlobalModelConfig = [
    ("FedAvg", "NumTrainingExamples", 1),
    ("FedAvg", "NumCompletedBatches", 1),
    ("FedAvg", "NumParticipants", 1),
    ("FedAvg", "NumTrainingExamples", 1),
]
LocalModelConfig = []
Controller = [] 
Learner = ["SSL"]
