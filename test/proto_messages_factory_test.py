import unittest

import numpy as np

from metisfl.proto.proto_messages_factory import ModelProtoMessages


class tensorsProtoTest(unittest.TestCase):

    def generate_and_validate_np_array(self, dtype):
        original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype)
        print(original_array)
        tensor = ModelProtoMessages.tensorsProto.numpy_array_to_proto_tensor_spec(
            original_array)
        print(tensor)
        print(tensor.value)
        converted_array = ModelProtoMessages.tensorsProto.proto_tensor_spec_to_numpy_array(
            tensor)
        # tests elements and shape equality
        self.assertTrue(np.array_equal(original_array, converted_array))
        self.assertTrue(original_array.dtype.byteorder ==
                        converted_array.dtype.byteorder)  # tests endian value

    def test_array_dtype_i1(self):
        self.generate_and_validate_np_array("i1")

    def test_array_dtype_i2(self):
        self.generate_and_validate_np_array("i2")

    def test_array_dtype_i4(self):
        self.generate_and_validate_np_array("i4")

    def test_array_dtype_i8(self):
        self.generate_and_validate_np_array("i8")

    def test_array_dtype_u1(self):
        self.generate_and_validate_np_array("u1")

    def test_array_dtype_u2(self):
        self.generate_and_validate_np_array("u2")

    def test_array_dtype_u4(self):
        self.generate_and_validate_np_array("u4")

    def test_array_dtype_u8(self):
        self.generate_and_validate_np_array("u8")

    def test_array_dtype_f4(self):
        self.generate_and_validate_np_array("f4")

    def test_array_dtype_f8(self):
        self.generate_and_validate_np_array("f8")


if __name__ == "__main__":
    unittest.main()
