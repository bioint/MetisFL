import datetime
from google.protobuf.timestamp_pb2 import Timestamp


def get_endpoint(hostname: str, port: int) -> str:
    return "{}:{}".format(hostname, port)


def get_timestamp() -> Timestamp:
    return Timestamp(seconds=int(
        datetime.datetime.now().timestamp()))
