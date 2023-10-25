"""
Test that the Python libraries are correctly installed by the Singularity container.
"""

import torch
import pytriton
import lightning
import neurobooth_analysis_tools
from gaze_analysis.model.backbone.uneye import UNEyeModule

print(f'GPU Enabled: {torch.cuda.is_available()}')
print(f'# GPUs: {torch.cuda.device_count()}')
print(f'PyTriton Version: {pytriton.__version__}')
