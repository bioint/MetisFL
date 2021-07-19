import os
import sys
import subprocess

from projectmetis.proto.controller_pb2 import ControllerParams, GlobalModelSpecs, CommunicationProtocolSpecs
from projectmetis.proto.shared_pb2 import ServerEntity


def cmd(args):
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = process.communicate()
    return out.decode('ascii').strip()


def default_controller_params():
    return ControllerParams(
        server_entity=ServerEntity(hostname='0.0.0.0', port=50051),
        global_model_specs=GlobalModelSpecs(learners_participation_ratio=1, aggregation_rule=GlobalModelSpecs.FED_AVG),
        communication_protocol_specs=CommunicationProtocolSpecs(protocol=CommunicationProtocolSpecs.SYNCHRONOUS),
        federated_execution_cutoff_mins=200,
        federated_execution_cutoff_score=0.85
    )