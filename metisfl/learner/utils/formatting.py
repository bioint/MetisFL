import json
import math

import pandas as pd


class DictionaryFormatter(object):

    @classmethod
    def normalize(cls, d):
        # Normalize dictionary to a flatten table.
        normalized = pd.json_normalize(d, sep="_")
        normalized = normalized.astype(str)
        normalized = normalized.to_dict(orient="records")[0]
        return normalized

    @classmethod
    def __stringify_val(cls, val):
        if isinstance(val, str):
            return val
        elif math.isnan(val):
            return "NaN"
        else:
            return str(val)

    @classmethod
    def stringify(cls, d, stringify_nan=True):
        tmp_d = dict()
        # One more pass to stringify nan values.
        for k, v in d.items():
            tmp_v = v
            if stringify_nan:
                if isinstance(v, list):
                    tmp_v = list(map(DictionaryFormatter.__stringify_val, v))
                else:
                    tmp_v = DictionaryFormatter.__stringify_val(v)
            tmp_d[k] = tmp_v
        stringified = \
            json.loads(json.dumps(tmp_d), parse_int=str, parse_float=str)
        return stringified

    @classmethod
    def listify_values(cls, d):
        d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
        return d
