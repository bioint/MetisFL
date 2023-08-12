GRPC_MAX_MESSAGE_LENGTH: int = 512 * 1024 * 1024


def get_endpoint(hostname: str, port: int) -> str:
    return "{}:{}".format(hostname, port)

