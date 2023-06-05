import random
import string


class DockerMetisServicesCmdFactory(object):

    def __init__(self, container_name=None):
        self.container_name = container_name
        if self.container_name is None:
            self.container_name = ''.join(random.SystemRandom().choice(
                string.ascii_uppercase + string.digits) for _ in range(10))

    def init_container(self,
                       port,
                       host_tmp_volume="/tmp/metis",
                       container_tmp_volume="/tmp/metis",
                       host_crypto_params_volume="/Users/Dstrip/CLionProjects/projectmetis-rc/resources/shelfi_cryptoparams",
                       container_crypto_params_volume="/metis/cryptoparams",
                       docker_image="projectmetis_rockylinux_8:0.0.1",
                       background_ps=True,
                       cuda_devices=None):

        docker_run_cmd = "docker run " \
                         "-p {port}:{port} " \
                         "-v {host_tmp_volume}:{container_tmp_volume} " \
                         "--name {container_name} "
        # "-v {host_crypto_params_volume}:{container_crypto_params_volume} " \

        if cuda_devices is not None:
            cuda_devices = "--gpus device={} ".format(','.join([str(d) for d in cuda_devices]))
            docker_run_cmd += cuda_devices
        if background_ps:
            docker_run_cmd += "-d "
        docker_run_cmd += "{docker_image}"
        docker_run_cmd = docker_run_cmd.format(
            port=port,
            host_tmp_volume=host_tmp_volume,
            container_tmp_volume=container_tmp_volume,
            host_crypto_params_volume=host_crypto_params_volume,
            container_name=self.container_name,
            container_crypto_params_volume=container_crypto_params_volume,
            docker_image=docker_image)
        return docker_run_cmd

    def attach_to_container(self):
        return "docker attach {}".format(self.container_name)

    def stop_container(self):
        return "docker stop {}".format(self.container_name)

    def rm_container(self):
        return "docker rm {}".format(self.container_name)
