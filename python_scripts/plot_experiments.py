from eos_line_search.run import *
from eos_line_search.plot import *
from eos_line_search.data import *
from haven import haven_utils as hu
import torch
import pickle
import io

from eos_line_search import plotting_experiments
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp",
    type=int,
    default=0,
    help="Mode of experiment to plot: 0 for main experiments, 1 for detailed diagnostics, 2 for NLS-ub, 3 for segment smoothness, 4 for stochastic, 5 for delta ablation, 6 for warmup, 7 for no bias",
)

args = parser.parse_args()

# Read arguments
exp = args.exp

if torch.cuda.is_available():
    print("Plotting with GPU")
else:
    print("Plotting with CPU")
loss = "mse"  # set to "mse" for mean squared error loss, or "ce" for cross-entropy loss

c = 1e-4


def plot_experiment(path):
    experiment_directory = os.path.join(path, "experiments")

    # Get all dataset directories
    dataset_dirs = [
        d
        for d in os.listdir(experiment_directory)
        if os.path.isdir(os.path.join(experiment_directory, d))
    ]

    for dataset in [
        "CIFAR10",
        "CIFAR100",
        "EMNIST",
        "SVHN",
    ]:
        dataset_path = os.path.join(experiment_directory, dataset)

        if os.path.exists(dataset_path) == False:
            continue

        # Get all model directories
        model_dirs = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]

        for model in [
            "CNN",
            "resnet34",
            "vgg11",
            "densenet121",
            "wide_resnet50_2",
            "MLP",
            "tinyVIT",
        ]:
            model_path = os.path.join(dataset_path, model)

            if os.path.exists(model_path) == False:
                continue

            # Get all batch_size directories
            batch_size_dirs = [
                d
                for d in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, d))
            ]

            for batch_size in ["full", "256"]:
                batch_size_path = os.path.join(model_path, batch_size)

                if os.path.exists(batch_size_path) == False:
                    continue

                # Get all optimizer directories
                optimizer_dirs = [
                    d
                    for d in os.listdir(batch_size_path)
                    if os.path.isdir(os.path.join(batch_size_path, d))
                ]

                runs_list = []
                for optimizer in optimizer_dirs:
                    optimizer_path = os.path.join(batch_size_path, optimizer)

                    # Get all pickle files in this directory
                    pickle_files = [
                        f
                        for f in os.listdir(optimizer_path)
                        if os.path.isfile(os.path.join(optimizer_path, f))
                    ]

                    # Load each pickle file
                    for pkl_file in pickle_files:
                        pkl_path = os.path.join(optimizer_path, pkl_file)

                        class CPUUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if (
                                    module == "torch.storage"
                                    and name == "_load_from_bytes"
                                ):
                                    return lambda b: torch.load(
                                        io.BytesIO(b), map_location="cpu"
                                    )
                                elif module.startswith("torch.cuda"):
                                    # Redirect CUDA classes to CPU equivalents
                                    if "FloatStorage" in name:
                                        return torch.FloatStorage
                                    elif "LongStorage" in name:
                                        return torch.LongStorage
                                    elif "IntStorage" in name:
                                        return torch.IntStorage
                                    elif "ByteStorage" in name:
                                        return torch.ByteStorage
                                    elif "DoubleStorage" in name:
                                        return torch.DoubleStorage
                                    elif "HalfStorage" in name:
                                        return torch.HalfStorage

                                return super().find_class(module, name)

                        if torch.cuda.is_available():
                            try:
                                result = hu.load_pkl(pkl_path)
                            except Exception as e:
                                print(f"Failed to load {pkl_path}: {e}")
                        else:
                            with open(pkl_path, "rb") as f:
                                result = CPUUnpickler(f).load()
                        run = result["run"]

                        runs_list.append(run)
                        print(f"Loaded: {pkl_path}")

                if not (batch_size == "full"):
                    if exp == 4:
                        plotting_experiments.plot_per_it(
                            runs_list,
                            path,
                            [
                                "Training Loss",
                                "Avg Batch Eigenvalue 1",
                                "Min Batch Eigenvalue 1",
                                "Step Size",
                                "Test Accuracy",
                            ],
                            exp,
                            loss,
                            c,
                        )
                    else:
                        print(f"Experiment {exp} not found")
                else:
                    if exp == 0 or exp == 2 or exp == 6:
                        plotting_experiments.plot_per_it(
                            runs_list,
                            path,
                            [
                                "Training Loss",
                                "Eigenvalue 1",
                                "Step Size",
                                "Test Accuracy",
                                "Sharpness x step",
                            ],
                            exp,
                            loss,
                            c,
                        )
                    elif exp == 5:
                        plotting_experiments.plot_per_it(
                            runs_list,
                            path,
                            [
                                "Training Loss",
                                "Function Evaluations",
                                "Eigenvalue 1",
                                "Step Size",
                                "Sharpness x step",
                            ],
                            exp,
                            loss,
                            c,
                        )
                    elif exp == 7:
                        plotting_experiments.plot_per_it(
                            runs_list,
                            path,
                            [
                                "Training Loss",
                                "Eigenvalue 1",
                                "Gradient Norm",
                            ],
                            exp,
                            loss,
                            c,
                        )
                    elif exp == 1:
                        plotting_experiments.plot_per_it(
                            runs_list,
                            path,
                            [
                                "Eigenvalue 1",
                                "Training Accuracy",
                                "Avg Hidden Grad Norm",
                                "Zero Grad Entries",
                                "Zero Activations",
                            ],
                            exp,
                            loss,
                            c,
                        )
                        plotting_experiments.plot_eigenvalues(runs_list, path, [], c)
                    elif exp == 3:
                        plotting_experiments.plot_assmpt_per_it(runs_list, path, exp)
                    else:
                        print(f"Experiment {exp} not found")


if __name__ == "__main__":
    plot_experiment(os.getcwd())
