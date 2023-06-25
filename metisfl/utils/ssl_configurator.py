# Most of this file content is taken from:
# https://unixutils.com/python-ssl-certificates-with-openssl/ #

import os
import yaml

import metisfl.utils.proto_messages_factory as proto_factory

from OpenSSL import crypto
from OpenSSL.crypto import \
    TYPE_RSA, TYPE_DSA, FILETYPE_PEM, load_certificate_request, PKCS12, FILETYPE_ASN1, load_privatekey, X509Req
from metisfl.proto.metis_pb2 import SSLConfig, SSLConfigFiles, SSLConfigStream
from metisfl.utils.metis_logger import MetisLogger


class SSLConfigurator(object):

    DIR = os.path.dirname(__file__)

    @classmethod
    def gen_default_certificates(cls, as_stream=False):
        # TODO(stripeli): Recheck if we need the default path certificates.
        public_certificate = os.path.join(
            cls.DIR, "../resources/ssl_config/default/server-cert.pem")
        private_key = os.path.join(
            cls.DIR, "../resources/ssl_config/default/server-key.pem")
        if as_stream:
            public_certificate = open(public_certificate, "rb").read()
            private_key = open(private_key, "rb").read()
        return public_certificate, private_key

    @classmethod
    def load_certificates_from_ssl_config_pb(cls, ssl_config_pb: SSLConfig, as_stream=False):
        public_certificate, private_key = None, None
        if ssl_config_pb.enable_ssl:
            ssl_config_attr = getattr(ssl_config_pb, ssl_config_pb.WhichOneof('config'))
            # If the certificate is given then establish secure channel connection.
            if isinstance(ssl_config_attr, SSLConfigFiles):
                public_certificate = ssl_config_pb.ssl_config_files.public_certificate_file
                private_key = ssl_config_pb.ssl_config_files.private_key_file
                if as_stream:
                    public_certificate = cls.load_file_as_stream(public_certificate)
                    private_key = cls.load_file_as_stream(private_key)
            elif isinstance(ssl_config_attr, SSLConfigStream):
                public_certificate = ssl_config_pb.ssl_config_stream.public_certificate_stream
                private_key = ssl_config_pb.ssl_config_stream.private_key_stream
            else:
                MetisLogger.warning("Even though SSL was requested the certificate "
                                    "was not provided. Proceeding without SSL.")

        return public_certificate, private_key

    @classmethod
    def load_file_as_stream(cls, filepath):
        stream = None
        if filepath and os.path.exists(filepath):
            stream = open(filepath, "rb").read()
        return stream

    @classmethod
    def gen_public_ssl_config_pb_as_stream(cls, ssl_config_pb: SSLConfig):
        config_pb = None
        if ssl_config_pb.enable_ssl:
            ssl_config_attr = getattr(ssl_config_pb, ssl_config_pb.WhichOneof('config'))
            stream = None
            if isinstance(ssl_config_attr, SSLConfigFiles):
                stream = \
                    cls.load_file_as_stream(ssl_config_pb.ssl_config_files.public_certificate_file)
            elif isinstance(ssl_config_attr, SSLConfigStream):
                stream = \
                    ssl_config_pb.ssl_config_stream.public_certificate_stream
            config_pb = proto_factory.MetisProtoMessages.construct_ssl_config_stream_pb(
                public_certificate_stream=stream)
        ssl_config_pb_aux = proto_factory.MetisProtoMessages.construct_ssl_config_pb(
            enable_ssl=ssl_config_pb.enable_ssl,
            config_pb=config_pb)
        return ssl_config_pb_aux
