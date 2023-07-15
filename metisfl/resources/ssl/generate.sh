#!/bin/bash

# Note: Running this file requires that openssl is installed on the target system.
rm *.pem

# 1. Generate Certificate Authorities' private key and self-signed certificate.
openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout ca-key.pem -out ca-cert.pem -subj "/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=NevronAI/emailAddress=metisfl@nevron.ai"

echo "Generating Server's certificate."
# 2. Generate server's private key and certificate signing request (CSR,-req.pem). CSR to be signed with CA's private key.
openssl req -newkey rsa:4096 -days 3600 -nodes -keyout server-key.pem -out server-req.pem -subj "/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=localhost/emailAddress=metisfl@nevron.ai"
# 3. Use Certificate Authorities' private key to sign web server's CSR and get back the signed certificate.
openssl x509 -req -in server-req.pem -days 3600 -CA ca-cert.pem -CAkey ca-key.pem -set_serial 01 -out server-cert.pem

echo "Generating Client's certificate."
openssl req -newkey rsa:4096 -days 3600 -nodes -keyout client-key.pem -out client-req.pem -subj "/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=localhost/emailAddress=metisfl@nevron.ai"
openssl x509 -req -in client-req.pem -days 3600 -CA ca-cert.pem -CAkey ca-key.pem -set_serial 01 -out client-cert.pem

echo "Validate Server's Signed certificate."
openssl x509 -in server-cert.pem -noout -tex
echo "Validate Clients's Signed certificate."
openssl x509 -in client-cert.pem -noout -tex