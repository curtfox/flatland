from dataclasses import dataclass


@dataclass
class Data:
    name: str
    train_subset: any = "full"
    n: int = 0
    d: int = 0
    image_height: int = 0
    image_width: int = 0
    image_colour_channels: int = 0
    no_test_set: bool = True
    one_hot_encode: bool = True
    input_dim: int = 0
    output_dim: int = 0
    label_avg: float = 0.0
    training_dataset: any = None
    testing_dataset: any = None
    stratified: bool = False
    centered: bool = False
