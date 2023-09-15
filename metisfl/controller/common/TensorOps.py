import struct
from typing import List

from google.protobuf.text_format import Parse
from metisfl.proto.model_pb2 import Tensor, TensorQuantifier


class TensorOps:
    """
    Generic tensor operations.
    """
    @staticmethod
    def deserialize_tensor(tensor: Tensor) -> List[float]:
        tensor_bytes = tensor.value
        tensor_elements_num = tensor.length
        deserialized_tensor = struct.unpack(f'{tensor_elements_num}d', tensor_bytes)
        return list(deserialized_tensor)

    @staticmethod
    def serialize_tensor(v: List[float]) -> bytes:
        serialized_tensor = struct.pack(f'{len(v)}d', *v)
        return serialized_tensor

    @staticmethod
    def quantify_tensor(tensor: Tensor) -> TensorQuantifier:
        t = TensorOps.deserialize_tensor(tensor)
        t_zeros = t.count(0)
        t_non_zeros = len(t) - t_zeros
        t_bytes = struct.calcsize('d') * len(t)

        tensor_quantifier = TensorQuantifier()
        tensor_quantifier.tensor_non_zeros = t_non_zeros
        tensor_quantifier.tensor_zeros = t_zeros
        tensor_quantifier.tensor_size_bytes = t_bytes

        return tensor_quantifier

    @staticmethod
    def print_serialized_tensor(data: bytes, num_values: int) -> None:
        loaded_values = struct.unpack(f'{num_values}d', data)
        print(', '.join(map(str, loaded_values)))

    @staticmethod
    def parse_text_or_die(input_str: str, message_type):
        result = message_type()
        Parse(input_str, result)
        return result
