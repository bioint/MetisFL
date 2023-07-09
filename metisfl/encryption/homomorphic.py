
from metisfl.encryption import fhe
from metisfl.models.types import ModelWeightsDescriptor
from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.proto_messages_factory import ModelProtoMessages

CRYPTO_RESOURCES_DIR = "/home/panoskyriakis/metisfl/metisfl/resources/fheparams/cryptoparams/"

class HomomorphicEncryption(object):

    def __init__(self, he_scheme_pb: metis_pb2.HEScheme):
        assert isinstance(
            he_scheme_pb, metis_pb2.HEScheme), "Not a valid HE scheme protobuf."
        
        if he_scheme_pb and he_scheme_pb.HasField("fhe_scheme"):
            self._he_scheme = fhe.CKKS(
                he_scheme_pb.fhe_scheme.batch_size,
                he_scheme_pb.fhe_scheme.scaling_bits,
                CRYPTO_RESOURCES_DIR)
            self._he_scheme.load_crypto_params()

    def decrypt_pb_weights(self,
                           variables: list[model_pb2.Model.Variable]) -> ModelWeightsDescriptor:
        assert all([isinstance(var, model_pb2.Model.Variable)
                   for var in variables])
        var_names, var_trainables, var_nps = list(), list(), list()
        for var in variables:
            # Variable specifications.
            var_name = var.name
            var_trainable = var.trainable

            if var.HasField("ciphertext_tensor"):
                assert self._he_scheme is not None, "Need encryption scheme to decrypt tensor."
                # For a ciphertext tensor, first we need to decrypt it, and then load it
                # into a numpy array with the data type specified in the tensor specifications.
                tensor_spec = var.ciphertext_tensor.tensor_spec
                tensor_length = tensor_spec.length
                decoded_value = self._he_scheme.decrypt(
                    tensor_spec.value, tensor_length, 1)
                # Since the tensor is decoded we just need to recreate the numpy array
                # to its original data type and shape.
                np_array = \
                    ModelProtoMessages.TensorSpecProto.proto_tensor_spec_with_list_values_to_numpy_array(
                        tensor_spec, decoded_value)
            elif var.HasField('plaintext_tensor'):
                tensor_spec = var.plaintext_tensor.tensor_spec
                # If the tensor is a plaintext tensor, then we need to read the byte buffer
                # and load the tensor as a numpy array casting it to the specified data type.
                np_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(
                    tensor_spec)
            else:
                raise RuntimeError("Not a supported tensor type.")

            # Append variable specifications to model's variable list.
            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(np_array)

        return ModelWeightsDescriptor(weights_names=var_names,
                                      weights_trainable=var_trainables,
                                      weights_values=var_nps)

    def encrypt_np_weights(self, weight_descriptor: ModelWeightsDescriptor) -> list[model_pb2.Model.Variable]:
        weights_names = weight_descriptor.weights_names
        weights_trainable = weight_descriptor.weights_trainable
        weights_values = weight_descriptor.weights_values
        if not weights_names:
            # Populating weights names with surrogate keys.
            weights_names = ["arr_{}".format(widx)
                             for widx in range(len(weights_values))]
        if weights_trainable:
            # Since weights have not specified as trainable or not, we default all weights to trainable.
            weights_trainable = [True for _ in range(len(weights_values))]

        variables_pb = []
        for w_n, w_t, w_v in zip(weights_names, weights_trainable, weights_values):
            ciphertext = None
            if self._he_scheme:
                ciphertext = self._he_scheme.encrypt(w_v.flatten(), 1)
            # If we have a ciphertext we prioritize it over the plaintext.
            tensor_pb = ModelProtoMessages.construct_tensor_pb(nparray=w_v,
                                                               ciphertext=ciphertext)
            model_var = ModelProtoMessages.construct_model_variable_pb(name=w_n,
                                                                       trainable=w_t,
                                                                       tensor_pb=tensor_pb)
            variables_pb.append(model_var)
        return variables_pb
