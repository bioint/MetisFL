import json
import math
import re

import pandas as pd


class DataTypeFormatter(object):

    @classmethod
    def camel_to_snake(cls, word):
        """
        Function taken from: 
        https://inflection.readthedocs.io/en/latest/#inflection.underscore
        Make an underscored, lowercase form from the expression in the word string.

        Example::

            >>> camel_to_snake_case("DeviceType")
            'device_type'

        """
        word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
        word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
        word = word.replace("-", "_")
        return word.lower()
    
    @classmethod
    def camel_to_snake_dict_keys(cls, d):
        return {cls.camel_to_snake(k): v for k, v in d.items()}

    @classmethod
    def normalize_dict(cls, d):
        # Normalize dictionary to a flatten table.
        normalized = pd.json_normalize(d, sep="_")
        normalized = normalized.astype(str)
        normalized = normalized.to_dict(orient="records")[0]
        return normalized

    @classmethod
    def stringify_dict(cls, d, stringify_nan=True):
        tmp_d = dict()
        # One more pass to stringify nan values.
        for k, v in d.items():
            tmp_v = v
            if stringify_nan:
                if isinstance(v, list):
                    tmp_v = list(map(DataTypeFormatter.__stringify_val, v))
                else:
                    tmp_v = DataTypeFormatter.__stringify_val(v)
            tmp_d[k] = tmp_v
        stringified = \
            json.loads(json.dumps(tmp_d), parse_int=str, parse_float=str)
        return stringified

    @classmethod
    def listify_dict_values(cls, d):
        d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
        return d
    
    @classmethod
    def __stringify_val(cls, val):
        if isinstance(val, str):
            return val
        elif math.isnan(val):
            return "NaN"
        else:
            return str(val)
