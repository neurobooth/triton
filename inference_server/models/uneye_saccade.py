import numpy as np
import torch
import pytriton.triton as triton
from pytriton.model_config import ModelConfig, Tensor, DynamicBatcher
from pytriton.decorators import batch
from typing import TypedDict

from inference_server.model import Model
from gaze_analysis.model.backbone.uneye import UNEyeModule


class ModelResult(TypedDict):
    proba: np.ndarray


class UNEyeSaccadeDetector(Model):
    def __init__(self, *args):
        super().__init__(*args)
        self.model = None
        self.device = None

    def load_model(self, weights_file: str, gpu: int) -> torch.nn.Module:
        """
        Load the model from a saved file.
        :param weights_file: The file containing saved model weights
        :param gpu: The GPU to run the model on.
        """
        self.device = torch.device('cuda', gpu)
        model = UNEyeModule.load_from_checkpoint(weights_file, map_location=self.device)
        torch.compile(model)
        model.eval()
        self.model = model.model
        return self.model

    @batch
    def infer(self, vel: np.ndarray) -> ModelResult:
        # self.logger.debug(f"Received batch of shape {vel.shape}")
        vel = torch.from_numpy(vel).to(self.device)  # Move input to GPU
        proba = self.model(vel)  # Run inference
        proba = proba.detach().cpu().numpy()  # Move result to CPU
        return {'proba': proba}

    def bind(self, triton: triton.Triton) -> None:
        triton.bind(
            model_name=self.model_name,
            infer_func=self.infer,
            inputs=[Tensor(name='vel', dtype=np.float32, shape=(2, -1))],
            outputs=[Tensor(name='proba', dtype=np.float32, shape=(2, -1))],
            config=ModelConfig(
                max_batch_size=128,
                batcher=DynamicBatcher(max_queue_delay_microseconds=int(1e4)),  # 10 ms
            ),
            strict=True,
        )
