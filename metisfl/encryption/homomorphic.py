from metisfl.utils.metis_logger import MetisLogger
from metisfl.encryption import fhe
from metisfl.proto import metis_pb2


class Homomorphic(object):

    def __init__(self, he_scheme_pb: metis_pb2.HESchemeConfig):
        """Initializes the Homomorphic object using the given HE scheme protobuf.

        **** ATTENTION ****
        If you modify the signature of this constructor, you must also modify the following accordingly:
        
        1. The constructor of the metisfl.learner.task_executor.TaskExecutor class.
        2. The functions metisfl.models.utils.{construct_model_pb, get_weights_from_model_pb}.
        3. The method metisfl.utils.fedenv.FederatedEnvironment.get_controller_he_scheme_pb.
        4. The state of the metisfl.driver.driver_session.DriverSession class.

        Args:
            he_scheme_pb (metis_pb2.HESchemeConfig): The protobuf object containing the HE scheme config.

        Raises:
            MetisLogger.fatal: If the given protobuf is not a valid HE scheme protobuf.
        """
        assert isinstance(
            he_scheme_pb, metis_pb2.HESchemeConfig), "Not a valid HE scheme protobuf."
        
        self._he_scheme = None        
        if he_scheme_pb.HasField("ckks_scheme_config"):
            self._he_scheme = self._construct_ckks_scheme(he_scheme_pb)
        elif he_scheme_pb.HasField("empty_scheme_config"):
            self._he_scheme = metis_pb2.EmptySchemeConfig()
        else:
            raise MetisLogger.fatal(
                "Not a supported HE scheme config. Received: {}".format(he_scheme_pb))

    @staticmethod
    def from_proto(he_scheme_pb: metis_pb2.HESchemeConfig):
        return Homomorphic(he_scheme_pb)

    def decrypt_data(self, ciphertext: str, num_elems: int):
        if isinstance(self._he_scheme, metis_pb2.EmptySchemeConfig):
            return None
        else:
            return self._he_scheme.decrypt(ciphertext, num_elems)

    def encrypt_data(self, values):
        if isinstance(self._he_scheme, metis_pb2.EmptySchemeConfig):
            return None
        else:                
            return self._he_scheme.encrypt(values)

    def _construct_ckks_scheme(self, he_scheme_pb: metis_pb2.HESchemeConfig):
        ckks_scheme = fhe.CKKS(he_scheme_pb.ckks_scheme_config.batch_size,
                               he_scheme_pb.ckks_scheme_config.scaling_factor_bits)
        if he_scheme_pb.crypto_context_file:
            ckks_scheme.load_crypto_context_from_file(he_scheme_pb.crypto_context_file)
        if he_scheme_pb.public_key_file:
            ckks_scheme.load_public_key_from_file(he_scheme_pb.public_key_file)
        if he_scheme_pb.private_key_file:
            ckks_scheme.load_private_key_from_file(he_scheme_pb.private_key_file)
        return ckks_scheme
