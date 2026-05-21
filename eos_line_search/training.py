from eos_line_search import optimizers_main
from eos_line_search.utils import *
import wandb
import torch.utils.data as torch_data
import torch as torch
from eos_line_search.experiment import *
from eos_line_search.run import *
from eos_line_search.plot import *
import math
from time import perf_counter


def train(run, device):
    run.model.model_obj.train()
    iteration = 0
    for epoch in range(0, run.epochs):
        print("--Epoch: ", epoch + 1)
        if run.batch_size == run.dataset.n:
            train_dataloader = torch_data.DataLoader(
                run.dataset.training_dataset,
                batch_size=run.batch_size,
                shuffle=False,
            )
        else:
            train_dataloader = torch_data.DataLoader(
                run.dataset.training_dataset, batch_size=run.batch_size, shuffle=True
            )
        start_time = perf_counter()
        if (
            "Average Batch Eigenvalues" in run.plot_metrics.metrics
            or "Max Batch Eigenvalues" in run.plot_metrics.metrics
            or "Min Batch Eigenvalues" in run.plot_metrics.metrics
        ):
            batch_eigenvalues = [[]] * run.plot_metrics.num_eigs
        for batch, (X, y) in enumerate(train_dataloader):
            print("--Iteration: ", iteration + 1)
            X = X.to(device)
            y = y.to(device)

            y = label_processing(run, y)

            frequency = run.plot_metrics.sharpness_every
            if (
                "Average Batch Eigenvalues" in run.plot_metrics.metrics
                or "Max Batch Eigenvalues" in run.plot_metrics.metrics
                or "Min Batch Eigenvalues" in run.plot_metrics.metrics
                and (epoch % frequency == 0 or epoch == run.epochs - 1)
            ):
                run.opt_obj.zero_grad()
                batch_eigenvalues = compute_batch_eigenvalues(
                    run, device, X, y, batch_eigenvalues
                )

            if batch == (run.num_batches - 1):
                current_metrics = {}
                if (
                    "Average Batch Eigenvalues" in run.plot_metrics.metrics
                    or "Max Batch Eigenvalues" in run.plot_metrics.metrics
                    or "Min Batch Eigenvalues" in run.plot_metrics.metrics
                ):
                    current_metrics["Batch Eigenvalues"] = batch_eigenvalues
                current_metrics = compute_metrics(run, device, epoch, current_metrics)
                if math.isnan(current_metrics["Training Loss"]) or math.isinf(
                    current_metrics["Training Loss"]
                ):
                    print("Invalid loss, stopping current run")
                    break

            if opt_line_search(run.optimizer.opt_name):
                run.opt_obj.zero_grad()

                def batch_loss_fn():
                    return compute_batch_training_loss_fn(run, X, y)

            elif run.optimizer.opt_name == "SAM":

                def batch_loss_fn():
                    run.opt_obj.zero_grad()
                    loss = compute_batch_training_loss_fn(run, X, y)
                    loss.backward()
                    return loss

            elif run.optimizer.opt_name == "CDAT":
                run.opt_obj.zero_grad()
                batch_loss_fn = compute_batch_training_loss_fn(run, X, y)
                batch_loss_fn.backward(retain_graph=True, create_graph=True)

            else:
                run.opt_obj.zero_grad()
                batch_loss_fn = compute_batch_training_loss_fn(run, X, y)
                batch_loss_fn.backward()

            # Update model parameters and compute new loss
            if batch == (run.num_batches - 1):
                (
                    current_metrics["Step Size"],
                    current_metrics["Initial Step Size"],
                    current_metrics["Backtracks"],
                    current_metrics["Function Evaluations"],
                    along_g_dict,
                ) = optimizers_main.opt_step(
                    batch_loss_fn,
                    run,
                    iteration,
                )
                time = perf_counter() - start_time
                current_metrics["time"] = time
                record_metrics(run, current_metrics, along_g_dict)

            else:
                optimizers_main.opt_step(
                    batch_loss_fn,
                    run,
                    iteration,
                )

            ### Record plot data on each epoch
            if batch == (run.num_batches - 1):
                # Compute BATCH gradient norm
                if "Gradient Norm" in run.plot_metrics.metrics:
                    current_metrics["Gradient Norm"] = compute_grad_norm(
                        run.model.model_obj.parameters()
                    )
                    run.plot_data["Gradient Norm"].append(
                        current_metrics["Gradient Norm"]
                    )
                    wandb.log(
                        {"Grad Norm": current_metrics["Gradient Norm"]}, commit=False
                    )
                    gradient_norm = current_metrics["Gradient Norm"]
                    print(f"Gradient Norm: {gradient_norm:.8f}")

                wandb.log({"Test": 0}, commit=True)

            iteration += 1
        else:
            continue
        break
    return run


def record_metrics(run, current_metrics, along_g_dict=None):
    if "time" in run.plot_metrics.metrics:
        run.plot_data["time"].append(current_metrics["time"])
        wandb.log(
            {"runtime": current_metrics["time"]},
            commit=False,
        )
    if "Backtracks" in run.plot_metrics.metrics:
        run.plot_data["Backtracks"].append(maybe_torch(current_metrics["Backtracks"]))
        wandb.log(
            {"# Line Search Steps": current_metrics["Backtracks"]},
            commit=False,
        )
    if "Function Evaluations" in run.plot_metrics.metrics:
        run.plot_data["Function Evaluations"].append(
            maybe_torch(current_metrics["Function Evaluations"])
        )
        wandb.log(
            {"# Function Evaluations": current_metrics["Function Evaluations"]},
            commit=False,
        )
    if "Step Size" in run.plot_metrics.metrics:
        run.plot_data["Step Size"].append(maybe_torch(current_metrics["Step Size"]))
        wandb.log(
            {"Step Size": current_metrics["Step Size"]},
            commit=False,
        )
    if "Initial Step Size" in run.plot_metrics.metrics:
        run.plot_data["Initial Step Size"].append(
            maybe_torch(current_metrics["Initial Step Size"])
        )
        wandb.log(
            {"Initial Step Size": current_metrics["Initial Step Size"]},
            commit=False,
        )
    if "Lw_asmpt" in run.plot_metrics.metrics:
        for key in along_g_dict:
            run.plot_data[key].append(along_g_dict[key])


def compute_metrics(run, device, epoch, current_metrics=None):

    # Compute FULL training loss and training accuracy
    if (
        "Training Loss" in run.plot_metrics.metrics
        or "Training Accuracy" in run.plot_metrics.metrics
    ):
        current_metrics["Training Loss"], current_metrics["Training Accuracy"] = (
            compute_full_training_loss(run, device)
        )
        run.plot_data["Training Loss"].append(
            maybe_torch(current_metrics["Training Loss"])
        )
        wandb.log({"Training Loss": current_metrics["Training Loss"]}, commit=False)
        run.plot_data["Training Accuracy"].append(
            maybe_torch(current_metrics["Training Accuracy"])
        )
        wandb.log(
            {"Training Accuracy": current_metrics["Training Accuracy"]}, commit=False
        )
        training_loss = current_metrics["Training Loss"]
        print(f"Training Loss: {training_loss:.8f}")
        training_accuracy = current_metrics["Training Accuracy"]
        print(f"Training Accuracy: {training_accuracy:.2f}")

    # Compute FULL test loss and test accuracy
    if (
        "Test Loss" in run.plot_metrics.metrics
        or "Test Accuracy" in run.plot_metrics.metrics
    ):
        current_metrics["Test Loss"], current_metrics["Test Accuracy"] = (
            compute_full_test_loss(run, device)
        )
        run.model.model_obj.train()
        run.plot_data["Test Loss"].append(maybe_torch(current_metrics["Test Loss"]))
        wandb.log({"Test Loss": current_metrics["Test Loss"]}, commit=False)
        run.plot_data["Test Accuracy"].append(
            maybe_torch(current_metrics["Test Accuracy"])
        )
        wandb.log({"Test Accuracy": current_metrics["Test Accuracy"]}, commit=False)
        test_loss = current_metrics["Test Loss"]
        print(f"Test Loss: {test_loss:.8f}")
        test_accuracy = current_metrics["Test Accuracy"]
        print(f"Test Accuracy: {test_accuracy:.2f}")

    # Compute eigenvalues
    run.opt_obj.zero_grad()
    frequency = run.plot_metrics.sharpness_every
    if "debugging" in run.plot_metrics.metrics and epoch >= run.plot_metrics.after_it:
        diagnose_training_issues(
            run,
            run.model.model_obj,
            torch_data.DataLoader(
                run.dataset.training_dataset, batch_size=run.batch_size, shuffle=False
            ),
            run.loss_fn,
            run.opt_obj,
        )

    if (
        "Eigenvalues" in run.plot_metrics.metrics
        and epoch >= run.plot_metrics.after_it
        and (epoch % frequency == 0 or epoch == run.epochs - 1)
    ):
        if "Sub-Hessian" in run.plot_metrics.metrics:
            # results = compute_eigenvalues(run, device, ['fc.weight', 'fc.bias']) #FIXME: these names are actually network-dependent
            # results = compute_eigenvalues(run, device, ["classifier.6.weight", "classifier.6.bias"])
            results = compute_eigenvalues(run, device, ["classifier.6.bias"])
        else:
            results = compute_eigenvalues(run, device)
        current_metrics["Eigenvalues"] = results.pop(0)

        if "Trace" in run.plot_metrics.metrics:
            trace = maybe_torch(results.pop(0))
            wandb.log({"Trace": trace}, commit=False)
            run.plot_data["Trace"].append(trace)

        for i, eig in enumerate(current_metrics["Eigenvalues"]):
            eig_val = maybe_torch(eig)
            run.plot_data["Eigenvalue " + str(i + 1)].append(eig_val)
            if i < 5:
                wandb.log({"Eigenvalue " + str(i + 1): eig_val}, commit=False)

            if "Perturbed Eigenvalues" in run.plot_metrics.metrics:
                if i == 0:
                    perturbed_eigs = results.pop(0)
                run.plot_data["Perturbed Eigenvalue " + str(i + 1)].append(
                    maybe_torch(perturbed_eigs[i])
                )
                if i < 5:
                    wandb.log(
                        {
                            "Perturbed Eigenvalue "
                            + str(i + 1): maybe_torch(perturbed_eigs[i])
                        },
                        commit=False,
                    )
        sharpness = current_metrics["Eigenvalues"][0]
        print(f"Sharpness: {sharpness:.8f}")

        if "PerturbedTrace" in run.plot_metrics.metrics:
            perturbed_trace = maybe_torch(results.pop(0))
            wandb.log({"PerturbedTrace": perturbed_trace}, commit=False)
            run.plot_data["PerturbedTrace"].append(perturbed_trace)

    # Compute BATCH eigenvalue metrics
    if "Avg Batch Eigenvalues" in run.plot_metrics.metrics and (
        epoch % frequency == 0 or epoch == run.epochs - 1
    ):
        batch_eigenvalues = current_metrics["Batch Eigenvalues"]
        avg_batch_eigenvalues = np.mean(batch_eigenvalues, axis=1)
        for i, eig in enumerate(avg_batch_eigenvalues):
            eig_val = maybe_torch(eig)
            run.plot_data["Avg Batch Eigenvalue " + str(i + 1)].append(eig_val)
            wandb.log({"Avg Batch Eigenvalue " + str(i + 1): eig_val}, commit=False)
        sharpness = avg_batch_eigenvalues[0]
        print(f"Avg Batch Sharpness: {sharpness:.8f}")

    if "Max Batch Eigenvalues" in run.plot_metrics.metrics and (
        epoch % frequency == 0 or epoch == run.epochs - 1
    ):
        batch_eigenvalues = current_metrics["Batch Eigenvalues"]
        max_batch_eigenvalues = np.max(batch_eigenvalues, axis=1)
        for i, eig in enumerate(max_batch_eigenvalues):
            eig_val = maybe_torch(eig)
            run.plot_data["Max Batch Eigenvalue " + str(i + 1)].append(eig_val)
            wandb.log({"Max Batch Eigenvalue " + str(i + 1): eig_val}, commit=False)
        sharpness = max_batch_eigenvalues[0]
        print(f"Max Batch Sharpness: {sharpness:.8f}")

    if "Min Batch Eigenvalues" in run.plot_metrics.metrics and (
        epoch % frequency == 0 or epoch == run.epochs - 1
    ):
        batch_eigenvalues = current_metrics["Batch Eigenvalues"]
        min_batch_eigenvalues = np.min(batch_eigenvalues, axis=1)
        for i, eig in enumerate(min_batch_eigenvalues):
            eig_val = maybe_torch(eig)
            run.plot_data["Min Batch Eigenvalue " + str(i + 1)].append(eig_val)
            wandb.log({"Min Batch Eigenvalue " + str(i + 1): eig_val}, commit=False)
        sharpness = min_batch_eigenvalues[0]
        print(f"Min Batch Sharpness: {sharpness:.8f}")

    return current_metrics
