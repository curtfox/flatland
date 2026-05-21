from dataclasses import dataclass


@dataclass
class Plot:
    metrics: list
    x_metric: str = "Iteration"
    label: str = "Optimizer"
    num_eigs: int = 1
    sharpness_every: int = 100
    after_it: int = 0
