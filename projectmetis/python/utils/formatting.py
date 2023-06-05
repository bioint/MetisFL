import json

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
    def stringify(cls, d):
        stringified = json.loads(json.dumps(d), parse_int=str, parse_float=str)
        return stringified

    @classmethod
    def listify_values(cls, d):
        d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
        return d
