import unittest
from unittest import mock
from google.protobuf.json_format import MessageToDict

from metisfl.common.types import TerminationSingals
from metisfl.driver.clients import GRPCControllerClient
from metisfl.driver.monitor import FederationMonitor
from metisfl.proto import controller_pb2


class TestFederationMonitor(unittest.TestCase):

    def setUp(self):
        self.client_mock = mock.Mock(spec=GRPCControllerClient)
        self.federation_monitor = FederationMonitor(
            termination_signals=TerminationSingals(
                evaluation_metric="accuracy",
                evaluation_metric_cutoff_score=0.9,
            ),
            controller_client=self.client_mock,
            is_async=True,
        )

        self.client_mock.get_logs.return_value = controller_pb2.Logs(
            global_iteration=1,
            tasks=[],
            train_results={},
            evaluation_results={},
            model_metadata={},
        )

    def tearDown(self):
        pass

    def test_monitor_federation(self):
        pass

    def test_reached_federation_rounds(self):
        pass

    def test_reached_evaluation_score(self):
        pass

    def test_reached_execution_time(self):
        pass

    def test_get_logs(self):
        logs = self.federation_monitor._get_logs()
        self.assertEqual(logs, self.client_mock.get_logs.return_value)

    def test_get_logs_dict(self):
        self.federation_monitor._get_logs()
        logs_dict = self.federation_monitor._get_logs_dict()
        self.assertEqual(
            logs_dict["global_iteration"],
            self.client_mock.get_logs.return_value.global_iteration
        )


if __name__ == '__main__':
    unittest.main()
