import torch
import pytriton.triton as triton

from inference_server.model import Model
from gaze_analysis.model.backbone.uneye import UNEyeModule


class UNEyeSaccadeDetector(Model):
    def __init__(self):
        self.model = None

    def load_model(self, weights_file: str, gpu: int) -> torch.nn.Module:
        """
        Load the model from a saved file.
        :param weights_file: The file containing saved model weights
        :param gpu: The GPU to run the model on.
        """
        model = UNEyeModule.load_from_checkpoint(weights_file, map_location=torch.device('cuda', gpu))
        torch.compile(model)
        model.eval()
        self.model = model.model
        return self.model

    def bind(self, triton: triton.Triton, model_name: str) -> None:
        print(model_name)
