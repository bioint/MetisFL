import abc

from collections import namedtuple
from metisfl.proto import model_pb2
from metisfl.utils.proto_messages_factory import ModelProtoMessages
from typing import List

ModelWeightsDescriptor = \
    namedtuple('ModelWeightsDescriptor',
               ['nn_engine', 'weights_names', 'weights_trainable', 'weights_values'])


def get_model_weights_from_variables_pb(variables: List[model_pb2.Model.Variable], he_scheme =  None):
    assert all([isinstance(var, model_pb2.Model.Variable) for var in variables])
    var_names, var_trainables, var_nps = list(), list(), list()
    for var in variables:
        # Variable specifications.
        var_name = var.name
        var_trainable = var.trainable

        if var.HasField("ciphertext_tensor"):
            assert he_scheme  is not None, "Need encryption scheme to decrypt tensor."
            # For a ciphertext tensor, first we need to decrypt it, and then load it
            # into a numpy array with the data type specified in the tensor specifications.
            tensor_spec = var.ciphertext_tensor.tensor_spec
            tensor_length = tensor_spec.length
            decoded_value = he_scheme.decrypt(tensor_spec.value, tensor_length, 1)
            # Since the tensor is decoded we just need to recreate the numpy array
            # to its original data type and shape.
            np_array = \
                ModelProtoMessages.TensorSpecProto.proto_tensor_spec_with_list_values_to_numpy_array(
                    tensor_spec, decoded_value)
        elif var.HasField('plaintext_tensor'):
            tensor_spec = var.plaintext_tensor.tensor_spec
            # If the tensor is a plaintext tensor, then we need to read the byte buffer
            # and load the tensor as a numpy array casting it to the specified data type.
            np_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(tensor_spec)
        else:
            raise RuntimeError("Not a supported tensor type.")

        # Append variable specifications to model's variable list.
        var_names.append(var_name)
        var_trainables.append(var_trainable)
        var_nps.append(np_array)

    # @stripeli what is var_nps? is this the same ase ModelWeightsDescriptor?
    return var_names, var_trainables, var_nps

