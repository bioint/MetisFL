
from metisfl.models.model_dataset import ModelDataset
from metisfl.utils.metis_logger import MetisLogger


def construct_dataset_pipeline(dataset: ModelDataset):
    _x = dataset.get_x()
    _y = dataset.get_y()
    if _x and _y:
        return _x, _y
    elif _x:
        return _x
    else:
        MetisLogger.error("Not a well-formatted input dataset: {}, {}".format(_x, _y))
        return None
