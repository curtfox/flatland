import torch
import copy
import numpy as np
import contextlib
from eos_line_search.utils import *


class LineSearch(torch.optim.Optimizer):
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
        params = list(params)
        super().__init__(params, {})
        self.params = params
        self.init_step_size = init_step_size
        self.max_eta = max_eta
        self.c = c
        self.beta = beta
        self.reset_option = reset_option
        self.forward_option = forward_option
        self.min_eta = min_eta
        self.n_batches_per_epoch = n_batches_per_epoch
        self.gamma = gamma
        self.c_p = c_p
        self.lk = 0
        self.save_backtracks = save_backtracks
        self.eps = eps
        self.state["step_size"] = init_step_size
        self.state["func_evals"] = 0
        self.state["grad_evals"] = 0
        self.state["backtracks"] = 0

    def step(self, closure, check_Lw_asmpt=False):
        self.state["backtracks"] = 0
        self.state["func_evals"] = 0
        seed = 0

        def closure_deterministic():
            with self.random_seed_torch(int(seed)):
                return closure()

        # get loss and compute gradients
        loss = closure_deterministic()
        self.state["func_evals"] += 1

        if check_Lw_asmpt or self.reset_option == 2:
            loss.backward(retain_graph=True, create_graph=True)
        else:
            loss.backward()

        self.state["grad_evals"] += 1

        # save the current parameters - compute once and reuse
        params_current = copy.deepcopy(self.params)
        grad_current = get_grad_list(self.params)
        grad_norm = compute_grad_norm(self.params)

        if check_Lw_asmpt:
            grad_fixed = get_grad_detached_list(self.params)

        # Reset step size
        step_size = self.reset_step_size(loss, self.state["step_size"], grad_norm)

        # Save step size before line search
        before_cut_step_size = step_size

        # Perform line search
        step_size, backtracks, function_evaluations = self.line_search(
            step_size,
            params_current,
            grad_current,
            loss,
            closure_deterministic,
            grad_norm,
        )

        # Compute lipschitz2, eigenvalue, and rayleigh for 10 different step sizes
        along_g_dict = {}
        if check_Lw_asmpt:
            along_g_dict = {
                "rayleigh": [],
                "lipschitz": [],
                "orig_lip": [],
                "eigen_val": [],
                "g_steps": [],
            }
            steps = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, step_size])
            sorted_ind = np.argsort(steps)
            for i, step in enumerate(steps):
                with torch.no_grad():
                    self.gd_update(self.params, step, params_current, grad_current)

                self.zero_grad()
                loss_temp = closure_deterministic()
                loss_temp.backward(retain_graph=True, create_graph=True)
                grad_temp = get_grad_list(self.params)

                # Computing Rayleigh Quotient along gradient
                hvp = torch.autograd.grad(
                    grad_temp,
                    tuple(self.params),
                    tuple(grad_current),
                    retain_graph=True,
                )
                rayleigh = compute_dot_product(hvp, grad_current) / (grad_norm**2)
                along_g_dict["rayleigh"].append(maybe_torch(rayleigh))

                # Computing lipschitz2 (Hessian norm estimate)
                lipschitz = compute_l2_norm(hvp) / grad_norm
                along_g_dict["lipschitz"].append(maybe_torch(lipschitz))

                orig_lip = compute_l2_norm(
                    subtract_lists(grad_fixed, grad_temp)
                ).item() / (step * grad_norm)
                along_g_dict["orig_lip"].append(orig_lip)

                # Computing dominant eigenvalue
                eigen_vec, eigen_val = self.extract_dominant_eigenvector(
                    closure_deterministic
                )
                along_g_dict["eigen_val"].append(maybe_torch(eigen_val))

                along_g_dict["g_steps"].append(maybe_torch(step))

                for param in self.params:
                    param.grad = None

                # Also detach intermediate results to break graph connections
                del loss_temp, grad_temp, hvp

            for key in along_g_dict.keys():
                along_g_dict[key] = np.array(along_g_dict[key])[sorted_ind].tolist()
            loss.backward()  # Not really needed, juet to plot "Gradient Norm" in training.py

        # Perform forward step
        self.state["step_size"] = self.forward_step(step_size)

        return (
            step_size,
            before_cut_step_size,
            backtracks,
            function_evaluations,
            along_g_dict,
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
        raise NotImplementedError

    def gd_update(self, params, step_size, params_current, grad_current):
        zipped = zip(params, params_current, grad_current)

        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - step_size * g_current

    def reset_step_size(self, loss, step_size, grad_norm):
        """
        Reset step size using different strategies including CDAT and MalMis

        Args:
            loss: current loss value
            step_size: current step size
            grad_norm: gradient norm
        """
        if self.reset_option == 0:
            step_size = max(min(step_size, self.max_eta), self.min_eta)

        elif self.reset_option == 1:
            polyak_step_size = loss / (self.c_p * grad_norm**2 + 1e-8)
            if self.save_backtracks:
                polyak_step_size = polyak_step_size * (self.beta**self.lk)
            step_size = max(min(polyak_step_size, self.max_eta), self.min_eta)

        return step_size

    def forward_step(self, step_size):

        if self.forward_option == 10:
            step_size = step_size

        return step_size

    # Armijo line search
    def check_armijo_conditions(
        self, step_size, loss, suff_dec, loss_next, c, beta, ref_value, eps
    ):
        found = 0
        # compute new break condition
        sufficient_decrease = (step_size) * c * suff_dec
        rhs = ref_value - sufficient_decrease + eps
        break_condition = loss_next - rhs
        if break_condition <= 0:
            found = 1
        else:
            # decrease the step-size by a multiplicative factor
            step_size = step_size * beta
            self.state["backtracks"] += 1

        return found, step_size, break_condition

    @contextlib.contextmanager
    def random_seed_torch(self, seed, device=0):
        cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            gpu_rng_state = torch.cuda.get_rng_state(0)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(gpu_rng_state, device)

    def compute_hessian_vector_product(self, loss, vector_list):
        """
        Computes the Hessian-vector product for a given loss function and a vector represented as a list of tensors.
        This uses the "double backprop" trick to avoid materializing the full Hessian.
        Args:
            loss: PyTorch scalar representing the loss
            vector_list: List of tensors with same shapes as model parameters
        Returns:
            Hessian-vector product as a list of tensors (same shapes as model parameters)
        """
        # First, compute gradients w.r.t. parameters
        grads = torch.autograd.grad(
            loss, tuple(self.params), create_graph=True, retain_graph=True
        )
        # Compute the gradient-vector dot product
        grad_vector_product = 0
        for grad, vector in zip(grads, vector_list):
            grad_vector_product += torch.sum(grad * vector)
        # Compute the gradient of this dot product w.r.t. the parameters
        hvp = torch.autograd.grad(
            grad_vector_product, tuple(self.params), retain_graph=True
        )
        # Return the Hessian-vector product as a list of tensors
        return list(hvp)

    def extract_dominant_eigenvector(self, loss_fn, num_iterations=100, tol=1e-3):
        """
        Computes the dominant eigenvector of the Hessian matrix using power iteration.
        Works with lists of tensors directly without flattening.
        Args:
            loss_fn: Loss function
            num_iterations: Maximum number of power iterations
            tol: Convergence tolerance
        Returns:
            dominant_eigenvector: The eigenvector corresponding to the largest eigenvalue as a list of tensors
            dominant_eigenvalue: The largest eigenvalue
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda")
        # Initialize a random vector list with same shapes as parameters
        vector_list = []
        for param in self.params:
            vector_list.append(torch.randn_like(param, device=device))
        # Normalize the vector list
        vector_norm = torch.sqrt(sum(torch.sum(v * v) for v in vector_list))
        vector_list = [v / vector_norm for v in vector_list]
        # Previous eigenvalue estimate for convergence check
        eigenvalue = None
        for i in range(num_iterations):
            # Reset gradients before forward pass
            self.zero_grad()
            # Forward pass
            loss = loss_fn()
            # Compute Hessian-vector product
            hvp_list = self.compute_hessian_vector_product(loss, vector_list)
            # Compute the Rayleigh quotient (dot product between hvp and vector)
            dot_product = sum(torch.sum(h * v) for h, v in zip(hvp_list, vector_list))
            tmp_eigenvalue = dot_product.item()
            # Calculate the norm of the hvp
            hvp_norm = torch.sqrt(sum(torch.sum(h * h) for h in hvp_list))
            # Normalize the vector for the next iteration
            vector_list = [h / hvp_norm for h in hvp_list]

            for param in self.params:
                param.grad = None
            # Check for convergence
            if eigenvalue is None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                    break
            eigenvalue = tmp_eigenvalue
        return vector_list, eigenvalue
