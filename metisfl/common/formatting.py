import datetime
import json
import math
import re

import pandas as pd
from google.protobuf.timestamp_pb2 import Timestamp


def get_endpoint(hostname: str, port: int) -> str:
    """Returns the endpoint string."""

    return "{}:{}".format(hostname, port)


def get_timestamp() -> Timestamp:
    """Returns the current timestamp as a protobuf Timestamp object."""

    return Timestamp(seconds=int(
        datetime.datetime.now().timestamp()))


def camel_to_snake(word):
    """
    Function taken from: 
    https://inflection.readthedocs.io/en/latest/#inflection.underscore
    Make an underscored, lowercase form from the expression in the word string.
    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    return word.lower()


def camel_to_snake_dict_keys(d):
    return {camel_to_snake(k): v for k, v in d.items()}


def normalize_dict(d):
    # Normalize dictionary to a flatten table.
    normalized = pd.json_normalize(d, sep="_")
    normalized = normalized.astype(str)
    normalized = normalized.to_dict(orient="records")[0]
    return normalized


def stringify_dict(d, stringify_nan=True):
    tmp_d = dict()
    # One more pass to stringify nan values.
    for k, v in d.items():
        tmp_v = v
        if stringify_nan:
            if isinstance(v, list):
                tmp_v = list(map(stringify_val, v))
            else:
                tmp_v = stringify_val(v)
        tmp_d[k] = tmp_v
    stringified = \
        json.loads(json.dumps(tmp_d), parse_int=str, parse_float=str)
    return stringified


def listify_dict_values(d):
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
    return d


def stringify_val(val):
    if isinstance(val, str):
        return val
    elif math.isnan(val):
        return "NaN"
    else:
        return str(val)
