from dataclasses import dataclass
from torch import nn


@dataclass
class Model:
    model_type: str
    model_obj: any = None
    model_num_params: int = 0
    bias: bool = True


@dataclass
class MLPModel(Model):
    activation_fn: any = nn.ReLU
    num_layers: int = 10
    width: int = 128


@dataclass
class CNNModel(Model):
    activation_fn: any = nn.ReLU
    pooling: any = nn.MaxPool2d
    window_size: int = 2
