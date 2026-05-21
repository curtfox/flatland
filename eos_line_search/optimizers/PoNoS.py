import torch
from eos_line_search.optimizers import LineSearch as LS
from eos_line_search.utils import *


class PoNoS(LS.LineSearch):
    """
    PoNoS Arguments:
         c=0.5, # line search sufficient decrease scaling constant
         c_p=0.1, # Polyak step size scaling constant
         delta=0.5, # cutting step
         zhang_xi=1, # Zhang xi, controlling the nonmonotonicity
         max_eta=10, # maximum step size
         min_eta=1e-06, #minimum step size
         f_star=0, # estimate of the min value of f
         save_backtracks=True # activate the memory-based resetting technique

         Note that PoNoS is like LBFGS from the LBFGS optimizer from pytorch,
         the step needs to be called like in the following:
         closure = lambda: loss_function(model, images, labels, backwards=False)
         opt.step(closure)
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
        zhang_xi=1,
        f_star=0,
        eps=0,
        num_classes=None,
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
        self.zhang_xi = zhang_xi
        self.state["Q_k"] = 0
        self.state["C_k"] = 0
        self.f_star = f_star
        # optional: number of classes for special power-of-two behavior
        self.num_classes = num_classes

    def _perform_armijo_line_search(
        self,
        step_size,
        params_current,
        grad_current,
        loss,
        closure_deterministic,
        suff_dec,
        ref_value,
    ):
        """
        Helper method to perform Armijo line search.
        Returns: (found, step_size, e)
        """
        for e in range(100):
            LS.LineSearch.gd_update(
                self, self.params, step_size, params_current, grad_current
            )
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
        return found, step_size, e

    def _handle_forward_options(
        self,
        step_size,
        params_current,
        grad_current,
        loss,
        closure_deterministic,
        suff_dec,
        ref_value,
    ):
        """
        Helper method to handle forward_option selection techniques.
        Returns: (step_size, found, e)
        """
        e = 0
        found = 1  # default: assume line search succeeded

        if self.forward_option == 10:
            while self.state["backtracks"] == 0:
                step_size = (1 / self.beta) * step_size
                found, step_size, e = self._perform_armijo_line_search(
                    step_size,
                    params_current,
                    grad_current,
                    loss,
                    closure_deterministic,
                    suff_dec,
                    ref_value,
                )

        elif self.forward_option == 11:
            while (
                self.state["backtracks"] == 0
                and ((1 / self.beta) * step_size) < self.num_classes
            ):
                step_size = (1 / self.beta) * step_size
                found, step_size, e = self._perform_armijo_line_search(
                    step_size,
                    params_current,
                    grad_current,
                    loss,
                    closure_deterministic,
                    suff_dec,
                    ref_value,
                )
                if found == 1:
                    break

        return step_size, found, e

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

            if grad_norm >= 1e-8 and loss.item() >= 1e-8:
                # check if condition is satisfied
                found = 0
                suff_dec = grad_norm**2

                # compute nonmonotone terms for the Zhang & Hager line search
                q_kplus1 = self.zhang_xi * self.state["Q_k"] + 1
                self.state["C_k"] = (
                    self.zhang_xi * self.state["Q_k"] * self.state["C_k"] + loss.item()
                ) / q_kplus1

                self.state["Q_k"] = q_kplus1
                ref_value = max(self.state["C_k"], loss.item())

                # perform line search
                found, step_size, e = self._perform_armijo_line_search(
                    step_size,
                    params_current,
                    grad_current,
                    loss,
                    closure_deterministic,
                    suff_dec,
                    ref_value,
                )

                ### new forward step selection techniques
                step_size, found, fwd_e = self._handle_forward_options(
                    step_size,
                    params_current,
                    grad_current,
                    loss,
                    closure_deterministic,
                    suff_dec,
                    ref_value,
                )

                if fwd_e is not None:
                    e = fwd_e

                # if line search exceeds 100 internal iterations
                if found == 0:
                    step_size = 1e-6
                    LS.LineSearch.gd_update(
                        self, self.params, step_size, params_current, grad_current
                    )

                self.lk = max(self.lk + e - 1, 0)

            else:
                print("Grad norm is {} and loss is {}".format(grad_norm, loss.item()))

        return step_size, self.state["backtracks"], self.state["func_evals"]
