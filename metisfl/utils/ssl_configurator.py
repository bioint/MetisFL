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
        public_certificate = os.path.join(
            cls.DIR, "../../../resources/ssl_config/default/server-cert.pem")
        private_key = os.path.join(
            cls.DIR, "../../../resources/ssl_config/default/server-key.pem")
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

    @classmethod
    def gen_self_signed_certificates(cls,
                                     files_prefix_name="test",
                                     certificate_dir="/tmp/ssl_config/",
                                     ssl_config_yaml=None):

        files_prefix_name = files_prefix_name
        certificate_dir = certificate_dir
        if not os.path.exists(certificate_dir):
            os.makedirs(certificate_dir)

        ssl_config_yaml = ssl_config_yaml
        if ssl_config_yaml is None:
            ssl_config_yaml = os.path.join(
                cls.DIR, "../../../resources/ssl_config/sample_config.yaml")
        try:
            loaded_configs = yaml.safe_load(open(ssl_config_yaml, "r"))
        except Exception as ConfigException:
            print("Failed to read Configuration %s" % ConfigException)

        if loaded_configs['CERT'].get('KeyType') == 'RSA':
            key_type = TYPE_RSA
        else:
            key_type = TYPE_DSA
        bit_length = loaded_configs['CERT'].get('BitLength')
        digest_type = loaded_configs['CERT'].get('DigestType')
        valid_from = loaded_configs['CERT'].get('ValidFrom')
        valid_to = loaded_configs['CERT'].get('ValidTo')

        # In case an existing private and certificate request file are provided then reuse them!
        existing_private_key = loaded_configs['REUSECONFIGS'].get('ReusePrivateKey')
        existing_private_key_type = loaded_configs['REUSECONFIGS'].get('ReusePrivateKeyType')
        existing_csr_file = loaded_configs["REUSECONFIGS"].get("ReuseCSRFile", None)
        existing_csr_file_type = loaded_configs["REUSECONFIGS"].get("ReuseCSRFileType", None)

        def gen_pkey():
            if not existing_private_key:
                pkey = crypto.PKey()
                pkey.generate_key(key_type, bit_length)
            else:
                with open(existing_private_key) as KeyFile:
                    if existing_private_key_type == 'PEM':
                        pkey = load_privatekey(FILETYPE_PEM, KeyFile.read())
                    elif existing_private_key_type == 'DER':
                        pkey = load_privatekey(FILETYPE_ASN1, KeyFile.read())
            return pkey

        def gen_csr(pkey):
            if not existing_csr_file:
                csr = X509Req()
                csr.get_subject().commonName = loaded_configs['CSR'].get('CommonName')
                csr.get_subject().countryName = loaded_configs['CSR'].get('CountryName')
                csr.get_subject().stateOrProvinceName = loaded_configs['CSR'].get('StateOrProvinceName')
                csr.get_subject().localityName = loaded_configs['CSR'].get('LocalityName')
                csr.get_subject().organizationName = loaded_configs['CSR'].get('OrganizationName')
                csr.get_subject().organizationalUnitName = loaded_configs['CSR'].get('OrganizationalUnitName')
                csr.get_subject().emailAddress = loaded_configs['CSR'].get('EmailAddress')
                csr.set_pubkey(pkey)
                csr.sign(pkey, digest_type)
            else:
                with open(existing_csr_file) as CsrFile:
                    if existing_csr_file_type == 'PEM':
                        csr = load_certificate_request(FILETYPE_PEM, CsrFile.read())
                    elif existing_csr_file_type == 'DER':
                        csr = load_certificate_request(FILETYPE_ASN1, CsrFile.read())
                    else:
                        raise TypeError("Unknown Certificate Type %s" % existing_csr_file_type)
            return csr

        def gen_cert(csr, pkey):
            cert = crypto.X509()
            cert.get_subject().commonName = csr.get_subject().commonName
            cert.get_subject().stateOrProvinceName = csr.get_subject().stateOrProvinceName
            cert.get_subject().localityName = csr.get_subject().localityName
            cert.get_subject().organizationName = csr.get_subject().organizationName
            cert.get_subject().organizationalUnitName = csr.get_subject().organizationalUnitName
            cert.get_subject().emailAddress = csr.get_subject().emailAddress
            cert.get_subject().countryName = csr.get_subject().countryName
            if valid_from and valid_to:
                cert.set_notBefore(valid_from)
                cert.set_notAfter(valid_to)
            cert.set_pubkey(pkey)
            cert.sign(pkey, digest_type)
            return cert

        def create_p12(pkey, cert, p12_file, passphrase=None):
            p12 = PKCS12()
            p12.set_certificate(cert)
            p12.set_privatekey(pkey)
            p12_file.write(p12.export(passphrase=passphrase))

        def gen_pkey_cert_csr():
            pkey = gen_pkey()
            csr = gen_csr(pkey)
            cert = gen_cert(csr, pkey)

            # Files in PEM format.
            pem_pkey_path = os.path.join(certificate_dir, files_prefix_name + "_pkey.pem")
            pem_csr_path = os.path.join(certificate_dir, files_prefix_name + "_csr.pem")
            pem_cert_path = os.path.join(certificate_dir, files_prefix_name + "_cert.pem")
            with open(pem_pkey_path, "wb+") as fPemPrivKey:
                fPemPrivKey.write(crypto.dump_privatekey(FILETYPE_PEM, pkey))
            with open(pem_csr_path, "wb+") as fPemcsr:
                fPemcsr.write(crypto.dump_certificate_request(FILETYPE_PEM, csr))
            with open(pem_cert_path, "wb+") as fPemcert:
                fPemcert.write(crypto.dump_certificate(FILETYPE_PEM, cert))

            # Files in DER (a.k.a. ASN1) format.
            der_pkey_path = os.path.join(certificate_dir, files_prefix_name + "_pkey.der")
            der_csr_path = os.path.join(certificate_dir, files_prefix_name + "_csr.der")
            der_cert_path = os.path.join(certificate_dir, files_prefix_name + "_cert.der")
            with open(der_pkey_path, "wb+") as fDerPrivKey:
                fDerPrivKey.write(crypto.dump_privatekey(FILETYPE_ASN1, pkey))
            with open(der_csr_path, "wb+") as fDercsr:
                fDercsr.write(crypto.dump_certificate_request(FILETYPE_ASN1, csr))
            with open(der_cert_path, "wb+") as fDercert:
                fDercert.write(crypto.dump_certificate(FILETYPE_ASN1, cert))

            # Files in p12/PFX format.
            p12_cert_path = os.path.join(certificate_dir, files_prefix_name + "_cert.p12")
            fp12_cert = open(p12_cert_path, "wb+")
            create_p12(pkey, cert, fp12_cert)
