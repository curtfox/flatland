from dataclasses import dataclass, asdict
from eos_line_search.plot import *
from eos_line_search.optim import *
from eos_line_search.model import *
from eos_line_search.data import *
from haven import haven_utils as hu
import os
import wandb
import numpy as np
import torch
from eos_line_search import optimizers_main, training

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.utils.data as torch_data
from collections import defaultdict
from torchvision import datasets
from eos_line_search.models import models
import random


@dataclass
class Run:
    dataset: Data
    loss_fn: any
    optimizer: Optim
    batch_size: any
    epochs: int
    model: Model
    plot_metrics: Plot
    num_batches: int = 0
    initialization: str = "default"
    use_bias: bool = True
    plot_data: dict[str, any] = None
    opt_obj: any = None
    run_id: str = ""

    def perform_run(
        self, use_wb, entity, project_name, group, experiment_directory, device, seed
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # run_id = hu.hash_dict({"run": self})
        run_id = self.get_run_id_with_exclusions()
        ### Save run_id in run dict
        self.run_id = run_id

        ### generate path for run
        run_dir = os.path.join(
            experiment_directory,
            self.dataset.name,
            self.model.model_type,
            str(self.batch_size),
            self.optimizer.opt_name,
        )
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_path = os.path.join(run_dir, run_id)

        if os.path.exists(run_path):
            print("Skipping ", run_path)
            return
        else:
            print(
                f"Running {run_id} on {self.dataset.name} x {self.model.model_type} via {self.optimizer.opt_name} with batch_size={self.batch_size} and epochs={self.epochs} with loss={self.loss_fn}"
            )
            print(self.optimizer)
            print(self.dataset)

        ### Process data
        self.process_data()

        ### Setup plotting for run
        self.plot_data = {}
        for metric in self.plot_metrics.metrics:
            if metric == "Eigenvalues":
                for i in range(self.plot_metrics.num_eigs):
                    self.plot_data["Eigenvalue " + str(i + 1)] = []
            elif metric == "Perturbed Eigenvalues":
                for i in range(self.plot_metrics.num_eigs):
                    self.plot_data["Perturbed Eigenvalue " + str(i + 1)] = []
            elif metric == "Avg Batch Eigenvalues":
                for i in range(self.plot_metrics.num_eigs):
                    self.plot_data["Avg Batch Eigenvalue " + str(i + 1)] = []
            elif metric == "Max Batch Eigenvalues":
                for i in range(self.plot_metrics.num_eigs):
                    self.plot_data["Max Batch Eigenvalue " + str(i + 1)] = []
            elif metric == "Min Batch Eigenvalues":
                for i in range(self.plot_metrics.num_eigs):
                    self.plot_data["Min Batch Eigenvalue " + str(i + 1)] = []
            elif metric == "Lw_asmpt":
                self.plot_data["rayleigh"] = []
                self.plot_data["lipschitz"] = []
                self.plot_data["orig_lip"] = []
                self.plot_data["eigen_val"] = []
                self.plot_data["g_steps"] = []
            else:
                self.plot_data[metric] = []

        ### Create model
        self.model.model_obj = models.select_model(self, device)
        self.model.model_num_params = sum(
            p.numel() for p in self.model.model_obj.parameters()
        )

        ### Set optimizer and loss function
        self.opt_obj = optimizers_main.setup_optimizer(self)

        if use_wb:
            if group is not None:
                group = (
                    group
                    + " "
                    + self.dataset.name
                    + " - "
                    + self.model.model_type
                    + " - "
                    + str(self.batch_size)
                    + " - "
                    + str(self.loss_fn)
                    + " - "
                    + str(self.epochs)
                )
            wandb.init(
                entity=entity,
                project=project_name,
                group=group,
                name=self.optimizer.opt_name
                + str(self.optimizer.forward_option)
                + "_"
                + str(self.optimizer.reset_option),
                config=asdict(self),
                settings=wandb.Settings(init_timeout=180),
            )
        else:
            wandb.init(mode="disabled")

        ### Train model
        print("---Training---")
        self = training.train(self, device)
        wandb.finish()

        ### Clear datasets before saving run
        self.dataset.training_dataset = None
        self.dataset.testing_dataset = None

        ### Clear model before saving run
        self.model.model_obj = None

        hu.save_pkl(run_path, {"run": self})

    def process_data(self):
        ### Image Datasets
        if (
            self.dataset.name == "CIFAR10"
            or self.dataset.name == "CIFAR100"
            or self.dataset.name == "SVHN"
            or self.dataset.name == "EMNIST"
        ):
            ### Training Set
            output_dim = 10
            if self.dataset.name == "CIFAR10":
                if not self.dataset.centered:
                    training_data = datasets.CIFAR10(
                        root="data",
                        train=True,
                        download=True,
                        transform=transforms.Compose([ToTensor()]),
                    )
                else:
                    training_data = datasets.CIFAR10(
                        root="data",
                        train=True,
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize(
                                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                                ),
                            ]
                        ),
                    )
            elif self.dataset.name == "CIFAR100":
                if not self.dataset.centered:
                    training_data = datasets.CIFAR100(
                        root="data",
                        train=True,
                        download=True,
                        transform=transforms.Compose([ToTensor()]),
                    )
                else:
                    training_data = datasets.CIFAR100(
                        root="data",
                        train=True,
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize(
                                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                                ),
                            ]
                        ),
                    )
                output_dim = 100
            elif self.dataset.name == "SVHN":
                if not self.dataset.centered:
                    training_data = datasets.SVHN(
                        root="data",
                        split="train",
                        download=True,
                        transform=ToTensor(),
                    )
                else:
                    training_data = datasets.SVHN(
                        root="data",
                        split="train",
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize(
                                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                                ),
                            ]
                        ),
                    )
            elif self.dataset.name == "EMNIST":
                if not self.dataset.centered:
                    training_data = datasets.EMNIST(
                        root="data",
                        split="letters",
                        train=True,
                        download=True,
                        transform=ToTensor(),
                    )
                else:
                    training_data = datasets.EMNIST(
                        root="data",
                        split="letters",
                        train=True,
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize((0.1722,), (0.3309,)),
                            ]
                        ),
                    )
                output_dim = 26

            # Stochastic setting
            if self.dataset.train_subset == "full":
                self.dataset.train_subset = len(training_data)

            validation_size = len(training_data) - self.dataset.train_subset
            if self.dataset.stratified:
                self.dataset.training_dataset, _ = stratified_random_split(
                    training_data,
                    [self.dataset.train_subset, validation_size],
                )
            else:
                self.dataset.training_dataset, _ = torch_data.random_split(
                    training_data,
                    [self.dataset.train_subset, validation_size],
                )

            self.dataset.n = len(self.dataset.training_dataset)

            if self.batch_size == "full":
                self.batch_size = self.dataset.n

            if (
                self.dataset.name == "CIFAR10"
                or self.dataset.name == "CIFAR100"
                or self.dataset.name == "SVHN"
            ):
                self.dataset.image_height = 32
                self.dataset.image_width = 32
                self.dataset.image_colour_channels = 3
            elif self.dataset.name == "EMNIST":
                self.dataset.image_height = 28
                self.dataset.image_width = 28
                self.dataset.image_colour_channels = 1

            self.set_training_data_parameters(
                torch_data.DataLoader(
                    self.dataset.training_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                ),
                True,
                self.dataset.image_height
                * self.dataset.image_width
                * self.dataset.image_colour_channels,
                output_dim,
            )

            ### Test Set
            self.dataset.no_test_set = False

            if self.dataset.name == "CIFAR10":
                if not self.dataset.centered:
                    test_data = datasets.CIFAR10(
                        root="data",
                        train=False,
                        download=True,
                        transform=transforms.Compose([ToTensor()]),
                    )
                else:
                    test_data = datasets.CIFAR10(
                        root="data",
                        train=False,
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize(
                                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                                ),
                            ]
                        ),
                    )
            elif self.dataset.name == "CIFAR100":
                if not self.dataset.centered:
                    test_data = datasets.CIFAR100(
                        root="data",
                        train=False,
                        download=True,
                        transform=ToTensor(),
                    )
                else:
                    test_data = datasets.CIFAR100(
                        root="data",
                        train=False,
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize(
                                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                                ),
                            ]
                        ),
                    )

            elif self.dataset.name == "SVHN":
                if not self.dataset.centered:
                    test_data = datasets.SVHN(
                        root="data",
                        split="test",
                        download=True,
                        transform=ToTensor(),
                    )
                else:
                    test_data = datasets.SVHN(
                        root="data",
                        split="test",
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize(
                                    (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
                                ),
                            ]
                        ),
                    )

            elif self.dataset.name == "EMNIST":
                if not self.dataset.centered:
                    test_data = datasets.EMNIST(
                        root="data",
                        split="letters",
                        train=False,
                        download=True,
                        transform=ToTensor(),
                    )
                else:
                    test_data = datasets.EMNIST(
                        root="data",
                        split="letters",
                        train=False,
                        download=True,
                        transform=transforms.Compose(
                            [
                                ToTensor(),
                                transforms.Normalize((0.1722,), (0.3309,)),
                            ]
                        ),
                    )

            test_subset = len(test_data)
            remaining_data = len(test_data) - test_subset
            self.dataset.testing_dataset, _ = torch_data.random_split(
                test_data, [test_subset, remaining_data]
            )

        else:
            raise ValueError("Not a valid dataset")

    def set_training_data_parameters(
        self, dataloader, one_hot_encode, input_dim, output_dim
    ):
        self.num_batches = len(dataloader)
        self.dataset.one_hot_encode = one_hot_encode
        self.dataset.input_dim = input_dim
        self.dataset.output_dim = output_dim

    def get_run_id_with_exclusions(self):
        """
        Generate a hash based only on primitive values (str, int, float, bool, None).
        Recursively extracts parameters from nested objects.
        """
        exclude_params = [
            "epochs",
            "plot_metrics",
            "num_batches",
            "plot_data",
            "opt_obj",
            "run_id",
        ]

        def extract_primitives(obj, prefix=""):
            """
            Recursively extract primitive values from an object.
            Returns a flat dictionary with keys like 'classname.paramname'.
            """
            result = {}
            # Handle primitive types directly
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            # Handle lists
            if isinstance(obj, list):
                return [extract_primitives(item, prefix) for item in obj]
            # Handle dictionaries
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}_{k}" if prefix else k
                    extracted = extract_primitives(v, new_prefix)
                    if isinstance(extracted, dict):
                        result.update(extracted)
                    else:
                        result[new_prefix] = extracted
                return result
            # Handle objects with __dict__ (dataclasses, custom classes)
            if hasattr(obj, "__dict__"):
                class_name = obj.__class__.__name__
                for k, v in obj.__dict__.items():
                    new_prefix = (
                        f"{prefix}_{class_name}_{k}" if prefix else f"{class_name}_{k}"
                    )
                    extracted = extract_primitives(v, new_prefix)
                    if isinstance(extracted, dict):
                        result.update(extracted)
                    else:
                        result[new_prefix] = extracted
                return result
            # For other types (torch types, etc.), convert to string
            return str(obj)

        # Build the filtered dictionary with primitives only
        flat_dict = {}
        for k, v in self.__dict__.items():
            if k not in exclude_params:
                extracted = extract_primitives(v, prefix=k)
                if isinstance(extracted, dict):
                    flat_dict.update(extracted)
                else:
                    flat_dict[k] = extracted
        #        for k in sorted(flat_dict.keys()):
        #            print(f"{k}: {flat_dict[k]}")
        run_id = hu.hash_dict(flat_dict)
        return run_id


def stratified_random_split(dataset, lengths, labels=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths,
    preserving class distributions (stratified sampling).

    Args:
        dataset: Dataset to be split
        lengths: Sequence of split sizes (can be fractions or integers)
        labels: Optional labels for stratification. If None, assumes dataset has 'targets' attribute

    Returns:
        List of Subset datasets
    """

    # Get labels
    if labels is None:
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        elif hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            # For ImageFolder-style datasets, try to get labels from samples
            if hasattr(dataset, "samples"):
                labels = [s[1] for s in dataset.samples]
            elif hasattr(dataset, "imgs"):  # Older ImageFolder versions
                labels = [s[1] for s in dataset.imgs]
            else:
                # Last resort: iterate through dataset
                labels = [dataset[i][1] for i in range(len(dataset))]

    # Convert to list if tensor
    if torch.is_tensor(labels):
        labels = labels.tolist()

    # Normalize lengths to fractions if they sum to dataset length
    total_length = len(dataset)
    if sum(lengths) == total_length:
        lengths = [l / total_length for l in lengths]
    elif abs(sum(lengths) - 1.0) > 1e-6:
        raise ValueError("Lengths must sum to 1.0 (as fractions) or to dataset length")

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Shuffle indices within each class
    for indices in class_indices.values():
        np.random.shuffle(indices)

    # Split indices for each class according to lengths
    split_indices = [[] for _ in lengths]

    for label, indices in class_indices.items():
        class_size = len(indices)
        start_idx = 0

        for i, length in enumerate(lengths[:-1]):  # All but last split
            split_size = int(class_size * length)
            split_indices[i].extend(indices[start_idx : start_idx + split_size])
            start_idx += split_size

        # Add remaining indices to last split
        split_indices[-1].extend(indices[start_idx:])

    # Shuffle the final splits
    for indices in split_indices:
        np.random.shuffle(indices)

    return [torch_data.Subset(dataset, indices) for indices in split_indices]
