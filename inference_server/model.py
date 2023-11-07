from abc import ABC, abstractmethod
import pytriton.triton as triton
import importlib
import torch.nn


class Model(ABC):
    """Base class that should be implemented by all served models"""

    @abstractmethod
    def load_model(self, weights_file: str, gpu: int) -> torch.nn.Module:
        """
        Load the model from a saved file.
        :param weights_file: The file containing saved model weights
        :param gpu: The GPU to run the model on.
        """
        raise NotImplementedError()

    @abstractmethod
    def bind(self, triton: triton.Triton, model_name: str) -> None:
        """
        Bind the model to the Triton server. (This method should call triton.bind() with the correct parameters.)
        :param triton: The Triron server object.
        :param model_name: The name of this model. (Should be passed to the model_name argument of bind().)
        """
        raise NotImplementedError()


def get_model_objects(class_name: str) -> Model:
    module, class_name = class_name.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)()
