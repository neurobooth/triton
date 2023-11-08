import os
import sys
from abc import ABC, abstractmethod
import pytriton.triton as triton
import importlib
import torch.nn
import logging


class Model(ABC):
    """Base class that should be implemented by all served models"""
    def __init__(self, model_name: str, log_dir: str):
        """
        :param model_name: The name of this model. (Should be passed to the model_name argument of bind().)
        :param log_dir: Directory to save logs to.
        """
        self.model_name = model_name
        self.logger = self._make_logger(log_dir)

    def _make_logger(self, log_dir: str) -> logging.Logger:
        """
        Create a logger object for this model.
        :param log_dir: Directory to save logs to.
        :return: The logger object.
        """
        logger = logging.getLogger(f'model.{self.model_name}')
        formatter = logging.Formatter(
            '|%(levelname)s| [%(asctime)s] %(filename)s, %(funcName)s, L%(lineno)d> %(message)s'
        )

        file_handler = logging.FileHandler(os.path.join(log_dir, f'model_{self.model_name}.log'), mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.setLevel(logging.DEBUG)
        return logger

    @abstractmethod
    def load_model(self, weights_file: str, gpu: int) -> torch.nn.Module:
        """
        Load the model from a saved file.
        :param weights_file: The file containing saved model weights
        :param gpu: The GPU to run the model on.
        """
        raise NotImplementedError()

    @abstractmethod
    def bind(self, triton: triton.Triton) -> None:
        """
        Bind the model to the Triton server. (This method should call triton.bind() with the correct parameters.)
        :param triton: The Triron server object.
        """
        raise NotImplementedError()


def get_model_objects(class_name: str, *args) -> Model:
    """
    Dynamically import and instantiate a model subclass based on the fully qualified class name.
    :param class_name: The fully qualified class name.
    :param args: Positional arguments to pass to the constructor during instantiation.
    :return: The Model object.
    """
    module, class_name = class_name.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)(*args)
