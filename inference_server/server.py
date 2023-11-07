import os
import sys
import logging
import yaml
from pydantic import BaseModel, DirectoryPath, FilePath
from typing import Dict

import torch
from pytriton.triton import Triton

from inference_server.model import Model, get_model_objects


class ModelConfig(BaseModel):
    class_name: str
    weights_file: FilePath
    gpu: int


class ServerConfig(BaseModel):
    log_dir: DirectoryPath
    models: Dict[str, ModelConfig]


def main() -> None:
    config = load_config('./config.yml')
    logger = setup_logging(config.log_dir)

    logger.info('Starting Server')
    check_torch()

    # Dynamically load model classes and create corresponding objects
    models: Dict[str, Model] = {
        name: get_model_objects(model_config.class_name) for name, model_config in config.models.items()
    }

    # Load model weights
    for name in models:
        logger.info(f'Loading {name}...')
        model, model_config = models[name], config.models[name]
        model = model.load_model(model_config.weights_file, model_config.gpu)
        model_size_mb = calc_model_size(model) / (1 << 20)
        logger.info(
            f'Loaded {name}, class={model.__class__.__module__}.{model.__class__.__name__}, size={model_size_mb:.1f} MiB'
        )

    with Triton() as triton:
        # Bind inference functions to request endpoints
        for name, model in models.items():
            logger.info(f'Binding {name}...')
            model.bind(triton, name)

        # Serve inference requests until Ctrl-C
        logger.info(f'Entering Request Loop')
        # triton.serve()


def load_config(file: str) -> ServerConfig:
    with open(file, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    return ServerConfig(**config)


def setup_logging(log_dir: str) -> logging.Logger:
    logger = logging.getLogger('server')
    formatter = logging.Formatter(
        '|%(levelname)s| [%(asctime)s] %(filename)s, %(funcName)s, L%(lineno)d> %(message)s'
    )

    file_handler = logging.FileHandler(os.path.join(log_dir, 'server.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
    return logger


def check_torch() -> None:
    log = logging.getLogger('server')
    if not torch.cuda.is_available():
        log.warning('CUDA is not available!')
        return

    n_gpus = torch.cuda.device_count()
    log.info(f'No. GPUs: {n_gpus}')
    for i in range(n_gpus):
        log.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')


def calc_model_size(model: torch.nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        size += buffer.nelement() * buffer.element_size()
    return size


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger = logging.getLogger('server')
        if logger.hasHandlers():
            logger.critical(e, exc_info=sys.exc_info())
        else:
            raise e
