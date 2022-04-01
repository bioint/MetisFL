## Standalone (Docker-Free) Prerequisites
- Install googletest (MacOS as `brew install googletest`)
- Install protobuf (MacOS as `brew install protobuf`)
- Run ./configure script 

## Bazel CLion comments 
If project files are not identifiable then you need to sync Bazel. To do so:
1. select the Bazel tab above
2. select the Sync subtab
3. and then Sync Project with BUILD Files

## Trello UI
https://trello.com/b/bYLUYqGK/metis-v01

## Docker
Due to some library inconsistencies that appeared across operating systems (e.g., Centos vs MacOC) we concluded that we 
should build the entire project within a docker container. The Dockerfile contains all the required setup.

To compile and run the project through docker, simply navigate to the parent directory of the project and then 
(assuming docker is already installed in the system):
    docker build -t projectmetis .

To run (or build) a bazel target (e.g., invoking the Homomorphic Encryption library through Python), we run:
```
docker run \ 
    -v /tmp/docker_metis_bazel:/tmp/docker_metis_bazel \
    -v /Users/Dstrip/CLionProjects/projectmetis-rc/resources/fhe_cryptoparams:/metis/cryptoparams \
    projectmetis \
    bazel \
        --output_user_root=/tmp/docker_metis_bazel \
        run //encryption/shelfi:shelfi_fhe_demo_main
```

In the above `docker run` example command we mount two volumes related to the output/build files of bazel and to the
path containing the directory of the cryptoparameters (e.g., public, private keys).
