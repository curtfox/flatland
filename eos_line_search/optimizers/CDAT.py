import torch
import copy
from eos_line_search.utils import *


class CDAT(torch.optim.Optimizer):
    def __init__(self, params, sigma, eps):
        params = list(params)
        super().__init__(params, {})
        self.params = params
        self.sigma = sigma
        self.eps = eps

    def get_step_size(self, params_current=None, grad_current=None, grad_norm=None):
        """
        Compute the CDAT step size without updating parameters

        Args:
            params_current: current parameters (optional, computed if not provided)
            grad_current: current gradients (optional, computed if not provided)
            grad_norm: current gradient norm (optional, computed if not provided)

        Returns:
            step_size: computed step size
        """
        if params_current is None:
            params_current = copy.deepcopy(self.params)
        if grad_current is None:
            grad_current = get_grad_list(self.params)
        if grad_norm is None:
            grad_norm = compute_grad_norm(self.params)

        # Compute Hessian-vector product
        hvp = torch.autograd.grad(
            tuple(grad_current), tuple(self.params), tuple(grad_current)
        )
        denom = torch.abs(compute_dot_product(hvp, grad_current)) + self.eps
        num = grad_norm**2
        step_size = self.sigma * (num / denom)

        return step_size

    def step(self):
        """Performs a single optimization step."""

        # save the current parameters
        params_current = copy.deepcopy(self.params)
        grad_current = get_grad_list(self.params)
        grad_norm = compute_grad_norm(self.params)

        step_size = self.get_step_size(params_current, grad_current, grad_norm)

        self.gd_update(self.params, step_size, params_current, grad_current)

        return step_size

    def gd_update(self, params, step_size, params_current, grad_current):
        zipped = zip(params, params_current, grad_current)

        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - step_size * g_current
