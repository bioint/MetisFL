import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import numpy as np
import torch
from torch import nn

from projectmetis.python.models.pytorch.pytorch_model_ops import PyTorchModelOps
from projectmetis.python.models.pytorch.quantization import (
    dequantize_tensor,
    quantize_tensor,
)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestPyTorchModelOps(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SimpleNet()
        self.ops = PyTorchModelOps(model=self.model)

    def test_get_model_weights_state_dict(self):
        weights = self.ops.get_model_weights()
        self.assertIsInstance(weights, dict)
        self.assertIn("fc1.weight", weights)
        self.assertIn("fc2.bias", weights)

    def test_get_model_weights_as_numpy(self):
        names, trainables, values = self.ops.get_model_weights(as_numpy=True)
        self.assertEqual(len(names), len(values))
        self.assertEqual(len(names), len(trainables))
        self.assertIn("fc1.weight", names)
        for v in values:
            self.assertIsInstance(v, np.ndarray)

    def test_get_model_weights_quantized(self):
        quant_dict = self.ops.get_model_weights(quantized=True, nbits=4, block_size=32)
        self.assertIn("layers", quant_dict)
        self.assertIn("fc1.weight", quant_dict["layers"])
        fc1_data = quant_dict["layers"]["fc1.weight"]
        self.assertEqual(fc1_data["type"], "quantized")
        self.assertEqual(fc1_data["packed"].dtype, torch.uint8)

    def test_set_model_weights_from_numpy(self):
        names, trainables, values = self.ops.get_model_weights(as_numpy=True)
        # Modify values
        new_values = [v + 1.0 for v in values]
        self.ops.set_model_weights(names, trainables, new_values)

        new_weights = self.ops.get_model_weights()
        self.assertTrue(
            torch.allclose(
                new_weights["fc1.weight"],
                torch.tensor(new_values[names.index("fc1.weight")]),
            )
        )

    def test_set_model_weights_from_quantized(self):
        quant_dict = self.ops.get_model_weights(quantized=True, nbits=4, block_size=32)
        # Should dequantize and load cleanly
        self.ops.set_model_weights(quant_dict)
        weights = self.ops.get_model_weights()
        self.assertIn("fc1.weight", weights)
        self.assertFalse(torch.isnan(weights["fc1.weight"]).any())

    def test_quantize_dequantize_tensor_roundtrip(self):
        t = torch.randn(10, 20, dtype=torch.float32)
        packed, scales, zeros, orig_shape, _ = quantize_tensor(
            t, nbits=4, block_size=32
        )
        recon = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=32
        )
        cos_sim = torch.cosine_similarity(t.flatten(), recon.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.95)


if __name__ == "__main__":
    unittest.main()
