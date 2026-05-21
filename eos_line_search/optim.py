from dataclasses import dataclass


@dataclass
class Optim:
    opt_name: str
    step_size: float = 0.0
    init_step_size: float = 0.0
    max_step_size: float = 0.0
    c: float = 0.0
    delta: float = 0.0
    reset_option: int = 0
    forward_option: int = 0
    eps: float = 0.0
    momentum: float = 0.0
    weight_decay: float = 0.0
    rho: float = 0.05
    xi: float = 1
