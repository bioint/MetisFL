import abc


class ModelDef:

    @abc.abstractmethod
    def get_model(self, *args, **kwargs):
        pass
