import torch
import torch.nn as nn
import os
from eos_line_search.experiment import *
from eos_line_search.data import *
from eos_line_search.run import *
from eos_line_search.plot import *
from eos_line_search.model import Model, CNNModel, MLPModel
from python_scripts import (
    optimizers,
    debug_optimizers,
    NLS_ub_optimizers,
    assmpt_optimizers,
    stochastic_optimizers,
    delta_ablation_optimizers,
    warmup_optimizers,
    no_bias_optimizers,
    best_step_size,
)
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset name")
parser.add_argument("--model", type=str, default="CNN", help="Model name")
parser.add_argument(
    "--batch_size", type=str, default="full", help="Batch size or full batch"
)
parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
parser.add_argument(
    "--mode",
    type=int,
    default=0,
    help="Mode of experiment to run: 0 for main experiments, 1 for detailed diagnostics, 2 for NLS-ub, 3 for segment smoothness, 4 for stochastic, 5 for delta ablation, 6 for warmup, 7 for no bias",
)
args = parser.parse_args()


# Read arguments
dataset = args.dataset
model = args.model
batch_size = args.batch_size
epochs = args.epochs
mode = args.mode

loss = "mse"
if loss == "mse":
    one_hot_encode = True
    loss_fn = nn.MSELoss(reduction="mean")
elif loss == "ce":
    one_hot_encode = False
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

use_bias = (
    True  # set to False to run experiments without bias terms, otherwise set to True
)
initialization = "default"

# Get Path
path = os.getcwd()

# Create folders for experiments and plots
if not os.path.exists(os.path.join(path, "experiments")):
    os.makedirs(os.path.join(path, "experiments"))

### Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### Setup plots and data subsetting
if batch_size == "full":
    train_subset = 5000
    plot_metrics = Plot(
        metrics=[
            "Training Loss",
            "Training Accuracy",
            "Eigenvalues",
            "Gradient Norm",
            "Step Size",
            "Function Evaluations",
            "Test Loss",
            "Test Accuracy",
        ],
        num_eigs=1,
        label="Optimizer",
        sharpness_every=100,
    )
else:
    train_subset = "full"
    plot_metrics = Plot(
        metrics=[
            "Training Loss",
            "Training Accuracy",
            "Avg Batch Eigenvalues",
            "Max Batch Eigenvalues",
            "Min Batch Eigenvalues",
            "Gradient Norm",
            "Step Size",
            "Function Evaluations",
            "Test Loss",
            "Test Accuracy",
        ],
        num_eigs=1,
        label="Optimizer",
        sharpness_every=2,
    )

if mode == 1:  # Detailed diagnostic experiments
    optimizers = debug_optimizers
    num_eigs = 20
    sharpness_every = 2
    if dataset == "EMNIST":
        num_eigs = 30
    after_it = 0
    if model == "vgg11" and dataset == "CIFAR10":
        after_it = 3200
    elif model == "vgg11" and dataset == "CIFAR100":
        after_it = 1200
    elif model == "vgg11" and dataset == "SVHN":
        after_it = 800
    elif model == "vgg11" and dataset == "EMNIST":
        after_it = 1100
    plot_metrics = Plot(
        metrics=[
            "Training Loss",
            "Training Accuracy",
            "Eigenvalues",
            "Gradient Norm",
            "Step Size",
            "Function Evaluations",
            "debugging",
            "Trace",
            "Bias Grad Norm",
            "Avg Hidden Grad Norm",
            "Std Hidden Grad Norm",
            "Min Hidden Grad Norm",
            "Max Hidden Grad Norm",
            "Zero Grad Entries",
            "Zero Activations",
            "Test Loss",
            "Test Accuracy",
        ],
        num_eigs=num_eigs,
        label="Optimizer",
        sharpness_every=sharpness_every,
        after_it=after_it,
    )

elif mode == 2:  # NLS-ub experiments
    optimizers = NLS_ub_optimizers

elif mode == 3:  # Segment smoothness experiments
    optimizers = assmpt_optimizers
    plot_metrics = Plot(
        metrics=[
            "Training Loss",
            "Training Accuracy",
            "Eigenvalues",
            "Gradient Norm",
            "Step Size",
            "Function Evaluations",
            "Test Loss",
            "Test Accuracy",
            "Lw_asmpt",
        ],
        num_eigs=1,
        label="Optimizer",
        sharpness_every=100,
    )

elif mode == 4:  # Stochastic experiments
    optimizers = stochastic_optimizers

elif mode == 5:  # Delta ablation experiments
    optimizers = delta_ablation_optimizers

elif mode == 6:  # Warmup experiments
    plot_metrics.sharpness_every = 1
    optimizers = warmup_optimizers

elif mode == 7:  # No bias experiments
    use_bias = False
    optimizers = no_bias_optimizers

### Setup runs
runs = []
epochs = epochs
if batch_size != "full":
    batch_size = int(batch_size)
for optimizer in optimizers:
    if optimizer.step_size == -1:
        maybe_step_size = best_step_size.get((optimizer.opt_name, dataset, model))
        if maybe_step_size:
            optimizer.step_size = maybe_step_size
            print("Best known step size: ", optimizer.step_size)
        else:
            optimizer.step_size = 0.1
    if model == "CNN":
        the_model = CNNModel(
            model_type="CNN", activation_fn=nn.ReLU, pooling=nn.MaxPool2d, window_size=2
        )
    elif model == "MLP":
        the_model = MLPModel(
            model_type="MLP", activation_fn=nn.ReLU, num_layers=3, width=100
        )
    else:
        the_model = Model(model_type=model)

    runs.append(
        Run(
            dataset=Data(
                name=dataset,
                train_subset=train_subset,
                stratified=True,
                one_hot_encode=one_hot_encode,
                centered=True,
            ),
            loss_fn=loss_fn,
            optimizer=copy.deepcopy(optimizer),
            batch_size=batch_size,
            epochs=epochs,
            model=the_model,
            plot_metrics=plot_metrics,
            initialization=initialization,
            use_bias=use_bias,
        )
    )

### Setup experiment
experiment = Experiment(runs=runs, device=device, path=path)

### Run experiment
use_wb = False  # set to True to use Weights and Biases logging
entity = None  # set Weights and Biases entity name, otherwise set to None
project_name = None  # set Weights and Biases project name, otherwise set to None
group = None  # set Weights and Biases group name, otherwise set to None
experiment.run_experiment(use_wb, entity, project_name, group)
