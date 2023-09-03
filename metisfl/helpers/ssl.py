""" Script to generate the SSL certificates for the server and client """

import os

CMDS = [
    "openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout ca-key.pem -out ca-cert.pem -subj \"/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=NevronAI/emailAddress=metisfl@nevron.ai\"",
    "openssl req -newkey rsa:4096 -days 3600 -nodes -keyout server-key.pem -out server-req.pem -subj \"/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=localhost/emailAddress=metisfl@nevron.ai\"",
    "openssl x509 -req -in server-req.pem -days 3600 -CA ca-cert.pem -CAkey ca-key.pem -set_serial 01 -out server-cert.pem",
    "openssl req -newkey rsa:4096 -days 3600 -nodes -keyout client-key.pem -out client-req.pem -subj \"/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=localhost/emailAddress=metisfl@nevron.ai\"",
    "openssl x509 -req -in client-req.pem -days 3600 -CA ca-cert.pem -CAkey ca-key.pem -set_serial 01 -out client-cert.pem",
    "openssl x509 -in server-cert.pem -noout -tex",
    "openssl x509 -in client-cert.pem -noout -tex",
]


def generate_ssl_certs():
    # chekf if openssl is installed using which
    if os.system("which openssl") != 0:
        print("openssl is not installed")
        exit(1)

    # generate the CA key and certificate
    for cmd in CMDS:
        os.system(cmd)


if __name__ == "__main__":
    generate_ssl_certs()
