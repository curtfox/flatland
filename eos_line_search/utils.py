import torch
import torch.nn as nn
import torch.utils.data as torch_data
from pyhessian import hessian
from eos_line_search.experiment import *
from eos_line_search.run import *
from eos_line_search.plot import *
import numpy as np


def compute_batch_training_loss_fn(run, X, y):
    pred = run.model.model_obj(X)
    loss = run.loss_fn(pred, y)
    return loss


def compute_batch_eigenvalues(run, device, X, y, batch_eigenvalues=None):
    # sum_eigenvalues = [0] * run.plot_metrics.num_eigs
    batch_hessian = hessian(
        run.model.model_obj,
        run.loss_fn,
        data=(X, y),
        cuda=use_cuda(device),
    )

    eigenvalues, _ = batch_hessian.eigenvalues(
        maxIter=100, tol=0.001, top_n=run.plot_metrics.num_eigs
    )
    batch_eigenvalues = [
        batch_eigenvalues[i] + [eigenvalues[i]] for i in range(len(batch_eigenvalues))
    ]
    return batch_eigenvalues


def compute_eigenvalues(run, device, layer_names=None):
    """
    Compute eigenvalues of the Hessian, optionally for specific layers.

    Args:
        run: Run object containing model, dataset, and configuration
        device: Device to run computations on
        layer_names: List of layer names to compute Hessians for (e.g., ['fc.weight', 'fc.bias'])
                    If None, computes full Hessian as before
    """
    for X, y in torch_data.DataLoader(
        run.dataset.training_dataset,
        batch_size=run.dataset.n,
        shuffle=False,
    ):
        X = X.to(device)
        y = y.to(device)
        y = label_processing(run, y)

        # Compute full Hessian
        full_hessian = hessian(
            run.model.model_obj,
            run.loss_fn,
            data=(X, y),
            cuda=use_cuda(device),
        )
        eigenvalues, _ = full_hessian.eigenvalues(
            maxIter=100, tol=0.001, top_n=run.plot_metrics.num_eigs
        )
        results = [eigenvalues]

        # If layer_names provided, compute layer-specific Hessians
        if layer_names is not None:
            print("\n" + "=" * 80)
            print("LAYER-WISE HESSIAN ANALYSIS")
            print("=" * 80)

            # Get parameter mapping
            param_dict = dict(run.model.model_obj.named_parameters())

            for layer_name in layer_names:
                if layer_name not in param_dict:
                    print(f"\nWarning: Layer '{layer_name}' not found in model")
                    continue

                param = param_dict[layer_name]
                param_size = param.numel()

                print(f"\n{'='*80}")
                print(f"Layer: {layer_name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Total parameters: {param_size}")
                print(f"{'='*80}")

                # Get layer-specific Hessian
                try:
                    layer_hessian = get_layer_hessian(
                        run.model.model_obj, run.loss_fn, X, y, layer_name, device
                    )

                    # Subsample if dimension > 100
                    if param_size > 100:
                        print(
                            f"Subsampling 100 dimensions uniformly from {param_size} parameters"
                        )
                        indices = np.linspace(0, param_size - 1, 100, dtype=int)
                        layer_hessian_sub = layer_hessian[np.ix_(indices, indices)]
                        print(f"Subsampled Hessian shape: {layer_hessian_sub.shape}")
                    else:
                        layer_hessian_sub = layer_hessian
                        print(f"Hessian shape: {layer_hessian_sub.shape}")
                        if "Sub-Hessian" in run.plot_metrics.metrics:
                            run.plot_data["Sub-Hessian"].append(layer_hessian)

                    # Print Hessian statistics
                    print(f"\nHessian Statistics:")
                    print(f"  Mean: {layer_hessian_sub.mean():.6e}")
                    print(f"  Std:  {layer_hessian_sub.std():.6e}")
                    print(f"  Min:  {layer_hessian_sub.min():.6e}")
                    print(f"  Max:  {layer_hessian_sub.max():.6e}")

                    # Compute eigenvalues
                    eigvals = np.linalg.eigvalsh(layer_hessian_sub)
                    eigvals = np.sort(eigvals)[::-1]  # Sort descending

                    print(f"\nEigenvalue Statistics:")
                    print(f"  Largest eigenvalue:  {eigvals[0]:.6e}")
                    print(f"  Smallest eigenvalue: {eigvals[-1]:.6e}")
                    print(f"  Condition number:    {abs(eigvals[0] / eigvals[-1]):.6e}")
                    print(f"  Trace:               {eigvals.sum():.6e}")

                    # Print top eigenvalues
                    n_print = min(10, len(eigvals))
                    print(f"\nTop {n_print} eigenvalues:")
                    for i in range(n_print):
                        print(f"  λ_{i+1}: {eigvals[i]:.6e}")

                    # Print Hessian matrix (or subset if too large)
                    print(f"\nHessian Matrix:")
                    if layer_hessian_sub.shape[0] <= 10:
                        print(layer_hessian_sub)
                    else:
                        print("First 10x10 block:")
                        print(layer_hessian_sub[:10, :10])

                except Exception as e:
                    print(f"Error computing Hessian for layer '{layer_name}': {str(e)}")
                    continue

        if "Trace" in run.plot_metrics.metrics:
            trace = full_hessian.trace()
            results.append(trace[0])

        if (
            "PerturbedTrace" in run.plot_metrics.metrics
            or "Perturbed Eigenvalues" in run.plot_metrics.metrics
        ):
            perturbation_scale = 0.001
            rand_data_list = []
            with torch.no_grad():
                for param in run.model.model_obj.parameters():
                    rand_data = torch.randn_like(param.data)
                    rand_data_list.append(rand_data)
                    param.data += perturbation_scale * rand_data

            h2 = hessian(
                run.model.model_obj,
                run.loss_fn,
                data=(X, y),
                cuda=use_cuda(device),
            )

            h2_eigenvalues, _ = h2.eigenvalues(
                maxIter=100, tol=0.001, top_n=run.plot_metrics.num_eigs
            )
            results.append(h2_eigenvalues)
            trace2 = h2.trace()
            results.append(trace2[0])
            for i, eig in enumerate(h2_eigenvalues):
                diff = abs(eigenvalues[i] - eig)
                print(
                    f" H λ_{i+1} = {eigenvalues[i]:.8f}, {eig:.8f}, diff = {diff:.2e}"
                )
            print(
                f"Traces: {trace[0]:.8f}, {trace2[0]:.8f}, diff = {abs(trace[0]-trace2[0]):.2e} with perturbation scale: {perturbation_scale}"
            )

            with torch.no_grad():
                for i, param in enumerate(run.model.model_obj.parameters()):
                    param.data -= perturbation_scale * rand_data_list[i]

    return results


def compute_full_training_loss(run, device):
    full_loss = 0
    correct = 0
    for X, y in torch_data.DataLoader(
        run.dataset.training_dataset,
        batch_size=1024,  # run.dataset.n,
        shuffle=False,
        # num_workers=4
    ):
        X = X.to(device)
        y = y.to(device)

        y_mod = label_processing(run, y)

        pred = run.model.model_obj(X)
        loss = run.loss_fn(pred, y_mod).item()
        full_loss = full_loss + loss * len(y)
        y_for_accuracy = y - 1 if run.dataset.name == "EMNIST" else y
        correct += (pred.argmax(1) == y_for_accuracy).type(torch.float).sum().item()

    full_loss = full_loss / run.dataset.n
    correct = 100 * (correct / run.dataset.n)

    return full_loss, correct


def compute_full_test_loss(run, device):
    test_n = len(run.dataset.testing_dataset)
    full_test_loss = 0
    correct = 0
    run.model.model_obj.eval()
    with torch.no_grad():
        for X, y in torch_data.DataLoader(
            run.dataset.testing_dataset,
            batch_size=1024,  # run.dataset.n,
            shuffle=False,
            # num_workers=4
        ):
            X = X.to(device)
            y = y.to(device)

            y_mod = label_processing(run, y)

            pred = run.model.model_obj(X)
            loss = run.loss_fn(pred, y_mod).item()
            full_test_loss = full_test_loss + loss * len(y)
            y_for_accuracy = y - 1 if run.dataset.name == "EMNIST" else y
            correct += (pred.argmax(1) == y_for_accuracy).type(torch.float).sum().item()

    full_test_loss = full_test_loss / test_n
    correct = 100 * (correct / test_n)

    return full_test_loss, correct


def get_grad_list(params):
    return [p.grad for p in params]


def compute_param_norm(parameters):
    params = [param.data.flatten() for param in parameters]
    params_cat = torch.cat(params)
    param_norm = torch.linalg.norm(params_cat, ord=2).item()
    return param_norm


def compute_grad_norm(parameters):
    grads = [param.grad.flatten() for param in parameters if param.grad is not None]
    grad_cat = torch.cat(grads)
    grad_norm = torch.linalg.norm(grad_cat, ord=2).item()
    return grad_norm


def compute_grad_inf_norm(parameters):
    grads = [param.grad.flatten() for param in parameters if param.grad is not None]
    grad_cat = torch.cat(grads)
    grad_norm = torch.linalg.norm(grad_cat, ord=float("inf")).item()
    return grad_norm


def opt_line_search(opt_name):
    # add here if more line search methods implemented
    return opt_name == "SLS" or opt_name == "PoNoS"


def compute_L_approximations(
    iteration, current_params, current_grad, prev_params=None, prev_grad=None
):
    # if first iteration compute placeholder value
    if iteration == 0:
        metric_7 = 0.0
        metric_8 = 0.0
        metric_9 = 0.0
    else:
        current_params = torch.cat(current_params)
        current_grad = torch.cat(current_grad)
        prev_params = torch.cat(prev_params)
        prev_grad = torch.cat(prev_grad)
        delta = current_params - prev_params
        y = current_grad - prev_grad
        metric_7 = torch.norm(y) / torch.norm(delta)
        metric_8 = torch.norm(y) ** 2 / torch.abs(torch.dot(delta, y))
        metric_9 = torch.abs(torch.dot(delta, y)) / torch.norm(delta) ** 2
    return metric_7, metric_8, metric_9


def compute_eig_density(current_hessian):
    eigenvalues, _ = current_hessian.density(iter=100, n_v=1)
    # density_plot_log(experiment, run, eigenvalues, weights)
    return np.array(eigenvalues), _


def maybe_torch(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value


def use_cuda(device):
    return device == "cuda"


def compute_dot_product(vect1_list, vect2_list):
    with torch.no_grad():
        dot_product = 0
        for v1, v2 in zip(vect1_list, vect2_list):
            dot_product += torch.sum(torch.mul(v1, v2))
    return dot_product


### probably need to refactor this at some point to make it more robust
def label_processing(run, y):
    if run.dataset.one_hot_encode == True:
        # if run.loss_fn == nn.MSELoss(reduction="mean"):
        if run.dataset.name == "EMNIST":
            y = y - 1
        y = nn.functional.one_hot(y, num_classes=run.dataset.output_dim).to(
            torch.float32
        )
        # else:
        #     raise ValueError("Not a valid loss function")
    else:
        #    y = y.unsqueeze(1).to(torch.float32)
        pass

    return y


def compute_l2_norm(list):
    with torch.no_grad():
        norm = 0.0
        for v in list:
            norm += torch.sum(torch.mul(v, v))
        norm = torch.sqrt(norm)
    return norm


def get_grad_detached_list(params):
    return [p.grad.clone().detach() for p in params]


def subtract_lists(list1, list2):
    to_return = []
    zipped = zip(list1, list2)
    with torch.no_grad():
        for l1, l2 in zipped:
            to_return.append(l1 - l2)
    return to_return


def test_diagonal_structure(hessian_comp, model, device, n_tests=20):
    """Test if Hessian is approximately diagonal"""

    # Create standard basis vectors (one-hot vectors)
    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)

    off_diagonal_norms = []
    diagonal_elements = []

    for i in range(min(n_tests, total_params)):
        # Create i-th standard basis vector
        e_i = torch.zeros(total_params).to(device)
        e_i[i] = 1.0

        # Reshape to match parameter structure
        e_i_shaped = []
        idx = 0
        for p in params:
            numel = p.numel()
            e_i_shaped.append(e_i[idx : idx + numel].reshape(p.shape))
            idx += numel

        # Compute H * e_i
        _, hv_product = hessian_comp.dataloader_hv_product(e_i_shaped)

        # Extract diagonal element (should be hv_product[i])
        hv_flat = torch.cat([h.flatten() for h in hv_product])
        diagonal_elements.append(hv_flat[i].item())

        # Measure off-diagonal elements
        hv_flat[i] = 0  # Zero out diagonal
        off_diagonal_norm = torch.norm(hv_flat).item()
        off_diagonal_norms.append(off_diagonal_norm)

    # Compute diagonality measure
    avg_off_diag = np.mean(off_diagonal_norms)
    avg_diag = np.mean(np.abs(diagonal_elements))

    diagonality_ratio = avg_off_diag / (avg_diag + 1e-10)
    print(f"Diagonality ratio (lower = more diagonal): {diagonality_ratio:.6f}")

    return diagonality_ratio < 0.1


def diagnose_training_issues(
    run, model, train_loader, criterion, optimizer, device="cuda"
):
    """
    Comprehensive diagnostic function to identify training problems.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function (automatically detects MSE vs CrossEntropy)
        optimizer: Optimizer
        device: Device to run on
    """

    model.train()

    # Automatically detect loss type
    loss_type = None
    if isinstance(criterion, nn.MSELoss):
        loss_type = "mse"
    elif isinstance(criterion, nn.CrossEntropyLoss):
        loss_type = "crossentropy"
    else:
        # Try to infer from class name
        criterion_name = type(criterion).__name__.lower()
        if "mse" in criterion_name or "l2" in criterion_name:
            loss_type = "mse"
        elif "cross" in criterion_name or "nll" in criterion_name:
            loss_type = "crossentropy"
        else:
            loss_type = "unknown"

    # Get one batch for testing
    data, target_raw = next(iter(train_loader))
    if run.dataset.name == "EMNIST":
        target_raw = target_raw - 1
    data, target_raw = data.to(device), target_raw.to(device)

    # Infer number of classes from data
    # Method 1: Check if targets are one-hot encoded
    if target_raw.dim() > 1 and target_raw.shape[1] > 1:
        num_classes = target_raw.shape[1]
    else:
        # Method 2: Get unique values from raw targets
        num_classes = len(target_raw.unique())
        # Method 3: Run a forward pass to get output dimension
        with torch.no_grad():
            temp_output = model(data[:1])
            if temp_output.shape[1] > num_classes:
                num_classes = temp_output.shape[1]

    # Process targets based on loss type
    if loss_type == "mse":
        # One-hot encode for MSE loss
        if target_raw.dim() == 1:
            target = nn.functional.one_hot(target_raw, num_classes=num_classes).to(
                torch.float32
            )
        else:
            target = target_raw.to(torch.float32)
        target_indices = (
            target_raw if target_raw.dim() == 1 else target_raw.argmax(dim=1)
        )
    elif loss_type == "crossentropy":
        # Use raw class indices
        if target_raw.dim() > 1:
            target = target_raw.argmax(dim=1)
        else:
            target = target_raw
        target_indices = target
    else:
        # Unknown loss type - try both formats
        print("   ⚠️  WARNING: Unknown loss type, will try to infer from shapes")
        target = target_raw
        target_indices = (
            target_raw if target_raw.dim() == 1 else target_raw.argmax(dim=1)
        )

    # Check for invalid target values and auto-fix
    target_min = target_indices.min().item()
    target_max = target_indices.max().item()

    # Auto-fix: Shift labels if they're out of range
    label_shift = 0
    if target_min < 0 or target_max >= num_classes:
        #    # Determine the shift needed
        if target_min > 0:
            label_shift = target_min
            target_indices = target_indices - label_shift
            # Also shift the one-hot encoded targets if using MSE
            if loss_type == "mse":
                target_raw_shifted = (
                    target_raw - label_shift
                    if target_raw.dim() == 1
                    else target_raw.argmax(dim=1) - label_shift
                )
                target = nn.functional.one_hot(
                    target_raw_shifted, num_classes=num_classes
                ).to(torch.float32)
            elif loss_type == "crossentropy":
                target = target_indices
        else:
            print(f"   ⚠️  ERROR: Negative target indices or complex label issue")
            print(
                f"   ⚠️  Range: [{target_min}, {target_max}], expected: [0, {num_classes-1}]"
            )
            return

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    grad_norms = {}
    zero_counts = {}
    total_counts = {}
    total_params_with_grad = 0
    total_zeros_grad = 0

    hidden_grad_norm_list = []

    for i, (name, param) in enumerate(model.named_parameters()):
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            hidden_grad_norm_list.append(grad_norm)

            zero_count = (param.grad == 0).sum().item()
            zero_counts[name] = zero_count
            total_counts[name] = param.grad.numel()
            total_zeros_grad += zero_count
            total_params_with_grad += param.grad.numel()

    # FIXME: a bit ugly, it assumes that the bias is in the last layer
    bias_grad_norm = hidden_grad_norm_list.pop()
    if "Bias Grad Norm" in run.plot_metrics.metrics:
        run.plot_data["Bias Grad Norm"].append(bias_grad_norm)

    # Print gradient statistics
    all_grads = list(grad_norms.values())
    if all_grads:

        if "Avg Hidden Grad Norm" in run.plot_metrics.metrics:
            run.plot_data["Avg Hidden Grad Norm"].append(
                np.nanmean(hidden_grad_norm_list)
            )
        if "Std Hidden Grad Norm" in run.plot_metrics.metrics:
            run.plot_data["Std Hidden Grad Norm"].append(
                np.nanstd(hidden_grad_norm_list)
            )
        if "Min Hidden Grad Norm" in run.plot_metrics.metrics:
            run.plot_data["Min Hidden Grad Norm"].append(
                np.nanmin(hidden_grad_norm_list)
            )
        if "Max Hidden Grad Norm" in run.plot_metrics.metrics:
            run.plot_data["Max Hidden Grad Norm"].append(
                np.nanmax(hidden_grad_norm_list)
            )

        zero_grad_pct = total_zeros_grad / total_params_with_grad * 100
        if "Zero Grad Entries" in run.plot_metrics.metrics:
            run.plot_data["Zero Grad Entries"].append(zero_grad_pct)

        # Print per-layer gradient norms (first and last few layers)
        items = list(grad_norms.items())
    else:
        print("   ⚠️  ERROR: No gradients found!")

    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                act = output.detach()
                activations[name] = {
                    "mean": act.mean().item(),
                    "std": act.std().item(),
                    "min": act.min().item(),
                    "max": act.max().item(),
                    "zeros": (act.abs() < 1e-6).float().mean().item() * 100,
                }

        return hook

    # Register hooks
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.BatchNorm2d)):
            hook = module.register_forward_hook(make_hook(f"{name}_{layer_idx}"))
            hooks.append(hook)
            layer_idx += 1

    # Forward pass to collect activations
    with torch.no_grad():
        _ = model(data)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print activation stats (first and last few)
    if activations:
        items = list(activations.items())
        max_zero_pct = 0
        for name, stats in items:
            if stats["zeros"] > max_zero_pct:
                max_zero_pct = stats["zeros"]
        if "Zero Activations" in run.plot_metrics.metrics:
            run.plot_data["Zero Activations"].append(max_zero_pct)


def get_layer_hessian(model, loss_fn, X, y, layer_name, device):
    """
    Compute Hessian matrix for a specific layer.

    Args:
        model: PyTorch model
        loss_fn: Loss function
        X: Input data
        y: Target labels
        layer_name: Name of the layer to compute Hessian for
        device: Device to run on

    Returns:
        Hessian matrix as numpy array
    """
    # Get the parameter
    param_dict = dict(model.named_parameters())
    param = param_dict[layer_name]
    param_size = param.numel()

    # Create a flat view of the parameter
    param_flat = param.view(-1)

    # Compute Hessian using autograd
    def compute_loss():
        model.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        return loss

    # Initialize Hessian matrix
    hessian_matrix = torch.zeros(param_size, param_size, device=device)

    # Compute Hessian via double backward pass
    for i in range(param_size):
        # First backward pass: compute gradient
        loss = compute_loss()
        grads = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[
            0
        ]
        grad_flat = grads.view(-1)

        # Second backward pass: compute Hessian row
        if grad_flat[i].requires_grad:
            model.zero_grad()
            grad_flat[i].backward(retain_graph=True)

            if param.grad is not None:
                hessian_matrix[i] = param.grad.view(-1).clone()
                param.grad.zero_()

    # Convert to numpy and make symmetric
    hessian_np = hessian_matrix.cpu().detach().numpy()
    hessian_np = (hessian_np + hessian_np.T) / 2

    return hessian_np


def is_decreasing_fast(values, threshold=0.1):
    """
    Check if values are decreasing rapidly, independent of magnitude.

    Parameters:
    -----------
    values : list or array
        The values to check (should be from sequential iterations)
    threshold : float
        Threshold for "fast" decrease. Default 0.1 means 10% decrease per iteration
        relative to the initial value.

    Returns:
    --------
    dict : Contains is_fast (bool), normalized_slope (float), and fit info
    """
    if len(values) < 2:
        return {
            "is_fast": False,
            "normalized_slope": 0,
            "error": "Need at least 2 values",
        }

    values = np.array(values)
    n = len(values)
    iterations = np.arange(n)

    # Fit a line: value = slope * iteration + intercept
    coeffs = np.polyfit(iterations, values, 1)
    slope, intercept = coeffs

    # Normalize slope by the first value (or mean, or max - your choice)
    # Using first value makes it represent "fraction of initial value lost per iteration"
    if abs(values[0]) < 1e-10:  # Avoid division by zero
        initial_value = (
            np.mean(np.abs(values)) if np.mean(np.abs(values)) > 1e-10 else 1
        )
    else:
        initial_value = abs(values[0])

    normalized_slope = slope / initial_value

    # For decreasing values, slope will be negative
    # Check if absolute value of normalized slope exceeds threshold
    is_fast = normalized_slope < -threshold

    print(
        f"Is fast? {is_fast}, Slope {slope}, Norm Slope {normalized_slope}, initial value {initial_value}"
    )
    return is_fast


def is_increasing_slowly(values, threshold=0.05, variance_threshold=0.05):
    """
    Check if values are increasing slowly with low variance, independent of magnitude.

    Parameters:
    -----------
    values : list or array
        The values to check (should be from sequential iterations)
    threshold : float
        Threshold for "slow" increase. Default 0.05 means 5% increase per iteration
        relative to the initial value is considered the upper bound for "slow".
    variance_threshold : float
        Threshold for normalized variance. Default 0.1 means variance should be less
        than 10% of the mean squared for a "slow and steady" increase.

    Returns:
    --------
    dict : Contains is_slow (bool), normalized_slope (float), variance info, and fit info
    """
    if len(values) < 2:
        return {
            "is_slow": False,
            "normalized_slope": 0,
            "error": "Need at least 2 values",
        }

    values = np.array(values)
    n = len(values)
    iterations = np.arange(n)

    # Fit a line: value = slope * iteration + intercept
    coeffs = np.polyfit(iterations, values, 1)
    slope, intercept = coeffs
    fitted_line = slope * iterations + intercept

    # Calculate residuals (deviations from the fitted line)
    residuals = values - fitted_line

    # Normalize slope by the first value (or mean if first is near zero)
    if abs(values[0]) < 1e-10:  # Avoid division by zero
        initial_value = (
            np.mean(np.abs(values)) if np.mean(np.abs(values)) > 1e-10 else 1
        )
    else:
        initial_value = abs(values[0])

    normalized_slope = slope / initial_value

    # Calculate normalized variance of residuals
    # This tells us how much the values deviate from a straight line
    mean_value = np.mean(values)
    if abs(mean_value) < 1e-10:
        mean_value = 1

    # Coefficient of variation of residuals
    residual_variance = np.var(residuals)
    normalized_variance = residual_variance / (mean_value**2)

    # Alternative: R-squared value (closer to 1 means better fit)
    ss_total = np.sum((values - np.mean(values)) ** 2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 1e-10 else 0

    # Check conditions for slow increase with low variance
    is_increasing_slowly = 0 < normalized_slope <= threshold
    has_low_variance = normalized_variance <= variance_threshold

    is_slow = is_increasing_slowly and has_low_variance

    print(
        f"Is slow? {is_increasing_slowly}, Has low variance? {has_low_variance}, Norm slope {normalized_slope}, Slope {slope}, Init val {initial_value}, Variance {residual_variance}, Norm Var {normalized_variance}, Mean {mean_value}"
    )
    return is_slow and (normalized_slope > 0)
