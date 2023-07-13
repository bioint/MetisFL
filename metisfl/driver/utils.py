
from metisfl.proto import proto_messages_factory
from metisfl.utils.fedenv import RemoteHost
from metisfl.utils.ssl_configurator import SSLConfigurator


def create_server_entity(enable_ssl: bool,
                         remote_host_instance: RemoteHost,
                         gen_connection_entity=False):

    # By default ssl is disabled.
    ssl_config_pb = None
    if enable_ssl:
        ssl_configurator = SSLConfigurator()
        if remote_host_instance.ssl_configs:
            # If the given instance has the public certificate and the private key defined
            # then we just wrap the ssl configuration around the files.
            wrap_as_stream = False
            public_cert = remote_host_instance.ssl_configs.public_certificate_file
            private_key = remote_host_instance.ssl_configs.private_key_file
        else:
            # If the given instance has no ssl configuration files defined, then we use
            # the default non-verified (self-signed) certificates, and we wrap them as streams.
            wrap_as_stream = True
            print("SSL enabled but remote host needs custom config files!", flush=True)
            public_cert, private_key = \
                ssl_configurator.gen_default_certificates(as_stream=True)

        if gen_connection_entity:
            # We only need to use the public certificate
            # to issue requests to the remote entity,
            # hence the private key is set to None.
            private_key = None

        if wrap_as_stream:
            ssl_config_bundle_pb = \
                proto_messages_factory.MetisProtoMessages.construct_ssl_config_stream_pb(
                    public_certificate_stream=public_cert,
                    private_key_stream=private_key)
        else:
            ssl_config_bundle_pb = \
                proto_messages_factory.MetisProtoMessages.construct_ssl_config_files_pb(
                    public_certificate_file=public_cert,
                    private_key_file=private_key)

        ssl_config_pb = \
            proto_messages_factory.MetisProtoMessages.construct_ssl_config_pb(
                enable_ssl=True,
                config_pb=ssl_config_bundle_pb)

    # The server entity encapsulates the GRPC servicer to which remote host will
    # spaw its grpc server and listen for incoming requests. It does not refer
    # to the connection configurations used to connect to the remote host.
    server_entity_pb = \
        proto_messages_factory.MetisProtoMessages.construct_server_entity_pb(
            hostname=remote_host_instance.grpc_hostname,
            port=remote_host_instance.grpc_port,
            ssl_config_pb=ssl_config_pb)
    return server_entity_pb
