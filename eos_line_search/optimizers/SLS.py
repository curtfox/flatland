import torch
from eos_line_search.optimizers import LineSearch as LS
from eos_line_search.utils import *


class SLS(LS.LineSearch):
    """Implements stochastic line search
    `paper <https://arxiv.org/abs/1905.09997>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        beta_b (float, optional): multiplicative factor for decreasing the step-size (default: 0.9)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        eta_max (float, optional): an upper bound used by Goldstein on the step size (default: 10)
    """

    def __init__(
        self,
        params,
        init_step_size=1,
        max_eta=10,
        c=0.5,
        beta=0.5,
        reset_option=0,
        forward_option=0,
        n_batches_per_epoch=500,
        gamma=2.0,
        min_eta=1e-06,
        c_p=0.1,
        save_backtracks=False,
        eps=0,
    ):
        super().__init__(
            params,
            init_step_size=init_step_size,
            max_eta=max_eta,
            c=c,
            beta=beta,
            reset_option=reset_option,
            forward_option=forward_option,
            n_batches_per_epoch=n_batches_per_epoch,
            gamma=gamma,
            min_eta=min_eta,
            c_p=c_p,
            save_backtracks=save_backtracks,
            eps=eps,
        )

    def line_search(
        self,
        step_size,
        params_current,
        grad_current,
        loss,
        closure_deterministic,
        grad_norm,
    ):
        with torch.no_grad():

            grad_norm = maybe_torch(grad_norm)

            if grad_norm >= 1e-8 and loss.item() != 0:
                # check if condition is satisfied
                found = 0
                suff_dec = grad_norm**2

                for e in range(100):
                    # try a prospective step
                    LS.LineSearch.gd_update(
                        self, self.params, step_size, params_current, grad_current
                    )

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    ref_value = loss.item()
                    self.state["func_evals"] += 1

                    found, step_size, _ = LS.LineSearch.check_armijo_conditions(
                        self=self,
                        step_size=step_size,
                        loss=loss.item(),
                        suff_dec=suff_dec,
                        loss_next=loss_next.item(),
                        c=self.c,
                        beta=self.beta,
                        ref_value=ref_value,
                        eps=self.eps,
                    )

                    if found == 1:
                        break

                    if step_size < 1e-6:
                        break

                if self.forward_option == 10:
                    while self.state["backtracks"] == 0:
                        # increase step size
                        step_size = 2 * step_size

                        # perform line search
                        for e in range(100):
                            # try a prospective step
                            LS.LineSearch.gd_update(
                                self,
                                self.params,
                                step_size,
                                params_current,
                                grad_current,
                            )

                            # compute the loss at the next step; no need to compute gradients.
                            loss_next = closure_deterministic()
                            self.state["func_evals"] += 1

                            found, step_size, _ = LS.LineSearch.check_armijo_conditions(
                                self=self,
                                step_size=step_size,
                                loss=loss.item(),
                                suff_dec=suff_dec,
                                loss_next=loss_next.item(),
                                c=self.c,
                                beta=self.beta,
                                ref_value=ref_value,
                                eps=self.eps,
                            )

                            if found == 1:
                                break

                # if line search exceeds max_epochs
                if found == 0:
                    step_size = 1e-6
                    LS.LineSearch.gd_update(
                        self, self.params, step_size, params_current, grad_current
                    )

                self.lk = max(self.lk + e - 1, 0)

            else:
                print("Grad norm is {} and loss is {}".format(grad_norm, loss.item()))

        return step_size, self.state["backtracks"], self.state["func_evals"]
