import numpy.lib.format
import sys

import numpy as np

from metisfl.proto import model_pb2


class ModelProtoMessages(object):

    class tensorsProto(object):

        NUMPY_DATA_TYPE_TO_PROTO_LOOKUP = {
            "i1": model_pb2.DType.Type.INT8,
            "i2": model_pb2.DType.Type.INT16,
            "i4": model_pb2.DType.Type.INT32,
            "i8": model_pb2.DType.Type.INT64,
            "u1": model_pb2.DType.Type.UINT8,
            "u2": model_pb2.DType.Type.UINT16,
            "u4": model_pb2.DType.Type.UINT32,
            "u8": model_pb2.DType.Type.UINT64,
            "f4": model_pb2.DType.Type.FLOAT32,
            "f8": model_pb2.DType.Type.FLOAT64
        }

        INV_NUMPY_DATA_TYPE_TO_PROTO_LOOKUP = {
            v: k for k, v in NUMPY_DATA_TYPE_TO_PROTO_LOOKUP.items()
        }

        @classmethod
        def numpy_array_to_proto_tensor_spec(cls, arr):

            # Examples of numpy arrays representation:
            #   "<i2" == (little-endian int8)
            #   "<u4" == (little-endian uint64)
            #   ">f4" == (big-endian float32)
            #   "=f2" == (system-default endian float8)
            # In general, the first character represents the endian type
            # and the subsequent characters the data type in the form of
            # integer(i), unsigned integer(u), float(f), complex(c) and the
            # digits the number of bytes, 4 refers to 4 bytes = 64bits.

            length = arr.size
            arr_metadata = numpy.lib.format.header_data_from_array_1_0(arr)
            shape = arr_metadata["shape"]
            dimensions = [s for s in shape]
            fortran_order = arr_metadata["fortran_order"]

            # For the byteorder representation in numpy check
            # https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
            descr = arr_metadata["descr"]
            if "<" in descr:
                endian = model_pb2.DType.ByteOrder.LITTLE_ENDIAN_ORDER
            elif ">" in descr:
                endian = model_pb2.DType.ByteOrder.BIG_ENDIAN_ORDER
            elif "=" in descr:
                endian = sys.byteorder
                if endian == "big":
                    endian = model_pb2.DType.ByteOrder.BIG_ENDIAN_ORDER
                else:
                    endian = model_pb2.DType.ByteOrder.LITTLE_ENDIAN_ORDER
            else:
                endian = model_pb2.DType.ByteOrder.NA  # case "|"

            nparray_dtype = descr[1:]
            if nparray_dtype in ModelProtoMessages.tensorsProto.NUMPY_DATA_TYPE_TO_PROTO_LOOKUP:
                proto_data_type = \
                    ModelProtoMessages.tensorsProto.NUMPY_DATA_TYPE_TO_PROTO_LOOKUP[
                        nparray_dtype]
            else:
                raise RuntimeError(
                    "Provided data type: {}, is not supported".format(nparray_dtype))

            dtype = model_pb2.DType(
                type=proto_data_type, byte_order=endian, fortran_order=fortran_order)

            flatten_array_bytes = arr.flatten().tobytes()
            tensor = model_pb2.Tensor(
                length=length, dimensions=dimensions, type=dtype, value=flatten_array_bytes)
            return tensor

        @classmethod
        def get_numpy_data_type_from_tensor_spec(cls, tensor):
            if tensor.type.byte_order == model_pb2.DType.ByteOrder.BIG_ENDIAN_ORDER:
                endian_char = ">"
            elif tensor.type.byte_order == model_pb2.DType.ByteOrder.LITTLE_ENDIAN_ORDER:
                endian_char = "<"
            else:
                endian_char = "|"

            data_type = tensor.type.type
            fortran_order = tensor.type.fortran_order
            np_data_type = \
                endian_char + \
                ModelProtoMessages.tensorsProto.INV_NUMPY_DATA_TYPE_TO_PROTO_LOOKUP[data_type]
            return np_data_type

        @classmethod
        def proto_tensor_spec_to_numpy_array(cls, tensor):
            np_data_type = \
                ModelProtoMessages.tensorsProto.get_numpy_data_type_from_tensor_spec(
                    tensor)
            dimensions = tensor.dimensions
            value = tensor.value
            length = tensor.length

            np_array = np.frombuffer(
                buffer=value, dtype=np_data_type, count=length)
            np_array = np_array.reshape(dimensions)

            return np_array

        @classmethod
        def proto_tensor_spec_with_list_values_to_numpy_array(cls, tensor, list_of_values):
            np_data_type = \
                ModelProtoMessages.tensorsProto.get_numpy_data_type_from_tensor_spec(
                    tensor)
            dimensions = tensor.dimensions

            np_array = np.array(list_of_values, dtype=np_data_type)
            np_array = np_array.reshape(dimensions)

            return np_array

    @classmethod
    def construct_tensor_pb(cls, nparray, ciphertext=None):
        # We prioritize the ciphertext over the plaintext.
        if not isinstance(nparray, np.ndarray):
            raise TypeError(
                "Parameter {} must be of type {}.".format(nparray, np.ndarray))

        tensor = \
            ModelProtoMessages.tensorsProto.numpy_array_to_proto_tensor_spec(
                nparray)

        if ciphertext is not None:
            # If the tensor is a ciphertext we need to set the bytes of the
            # ciphertext as the value of the tensor not the numpy array bytes.
            tensor.value = ciphertext
            tensor_pb = model_pb2.CiphertextTensor(tensor=tensor)
        else:
            tensor_pb = model_pb2.PlaintextTensor(tensor=tensor)
        return tensor_pb

    @classmethod
    def construct_model_variable_pb(cls, name, trainable, tensor_pb):
        assert isinstance(name, str) and isinstance(trainable, bool)
        if isinstance(tensor_pb, model_pb2.PlaintextTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, plaintext_tensor=tensor_pb)
        elif isinstance(tensor_pb, model_pb2.CiphertextTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, ciphertext_tensor=tensor_pb)
        else:
            raise RuntimeError(
                "Tensor proto message refers to a non-supported tensor protobuff datatype.")

    @classmethod
    def construct_model_pb_from_vars_pb(
            cls, variables_pb):
        assert isinstance(variables_pb, list) and \
            all([isinstance(var, model_pb2.Model.Variable)
                for var in variables_pb])
        return model_pb2.Model(variables=variables_pb)

    @classmethod
    def construct_federated_model_pb(cls, num_contributors, model_pb):
        assert isinstance(model_pb, model_pb2.Model)
        return model_pb2.Model(num_contributors=num_contributors, model=model_pb)

    @classmethod
    def construct_optimizer_config_pb(cls, optimizer_name, learning_rate, kwargs):
        return model_pb2.OptimizerConfig(
            name=optimizer_name, learning_rate=learning_rate, kwargs=kwargs)
