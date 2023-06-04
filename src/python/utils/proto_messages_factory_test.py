import unittest

import numpy as np

from src.python.utils.proto_messages_factory import ModelProtoMessages


class TensorSpecProtoTest(unittest.TestCase):

    def _generate_and_validate_np_array(self, dtype):
        original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype)
        print(original_array)
        tensor_spec = ModelProtoMessages.TensorSpecProto.numpy_array_to_proto_tensor_spec(original_array)
        print(tensor_spec)
        print(tensor_spec.value)
        converted_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(tensor_spec)
        self.assertTrue(np.array_equal(original_array, converted_array))  # tests elements and shape equality
        self.assertTrue(original_array.dtype.byteorder == converted_array.dtype.byteorder)  # tests endian value

    def test_array_dtype_i1(self):
        self._generate_and_validate_np_array("i1")

    def test_array_dtype_i2(self):
        self._generate_and_validate_np_array("i2")

    def test_array_dtype_i4(self):
        self._generate_and_validate_np_array("i4")

    def test_array_dtype_i8(self):
        self._generate_and_validate_np_array("i8")

    def test_array_dtype_u1(self):
        self._generate_and_validate_np_array("u1")

    def test_array_dtype_u2(self):
        self._generate_and_validate_np_array("u2")

    def test_array_dtype_u4(self):
        self._generate_and_validate_np_array("u4")

    def test_array_dtype_u8(self):
        self._generate_and_validate_np_array("u8")

    def test_array_dtype_f4(self):
        self._generate_and_validate_np_array("f4")

    def test_array_dtype_f8(self):
        self._generate_and_validate_np_array("f8")


if __name__ == "__main__":
    unittest.main()
