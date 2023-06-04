#!/bin/bash

# Note: Running this file requires that openssl is installed on the target system.
rm *.pem

# 1. Generate Certificate Authorities' private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout ca-key.pem -out ca-cert.pem -subj "/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=localhost/emailAddress=src.rc@gmail.com"

echo "Certificate Authorities''s self-signed certificate"
openssl x509 -in ca-cert.pem -noout -text

# 2. Generate web server's private key and certificate signing request (CSR). CSR to be signed with CA's private key
openssl req -newkey rsa:4096 -nodes -keyout server-key.pem -out server-req.pem -subj "/C=US/ST=CA/L=LosAngeles/O=ISI/OU=Research/CN=localhost/emailAddress=src.rc@gmail.com"

# 3. Use Certificate Authorities' private key to sign web server's CSR and get back the signed certificate
openssl x509 -req -in server-req.pem -days 60 -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem

echo "Server's signed certificate"
openssl x509 -in server-cert.pem -noout -text