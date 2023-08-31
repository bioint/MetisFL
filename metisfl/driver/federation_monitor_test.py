import unittest

from metisfl.proto import controller_pb2

from metisfl.common.types import TerminationSingals
from metisfl.driver.controller_client import GRPCControllerClient
from metisfl.driver.federation_monitor import FederationMonitor

# write the sceleton of the test class


class TestFederationMonitor(unittest.TestCase):

    def setUp(self):
        client_mock = unittest.mock.Mock(spec=GRPCControllerClient)
        signals_mock = unittest.mock.Mock(TerminationSingals)
        self.federation_monitor = FederationMonitor(
            client=client_mock,
            signals=signals_mock,
            is_active=True,
        )

        client_mock.get_logs.return_value = controller_pb2.Logs(
            global_iteration=1,
            task_learner_map={},
            train_results={},
            evaluation_results={},
            model_results={},
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

    def test__get_logs(self):
        client_mock = unittest.mock.Mock(spec=GRPCControllerClient)
        signals_mock = unittest.mock.Mock(TerminationSingals)
        federation_monitor = FederationMonitor(
            client=client_mock,
            signals=signals_mock,
            is_active=True,
        )
        logs = federation_monitor._get_logs()
        self.assertEqual(logs, client_mock.get_logs.return_value)


if __name__ == '__main__':
    unittest.main()
