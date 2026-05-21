import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import copy
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch.nn as nn


def noneIsInfinite(value):
    if value is None:
        return float("inf")
    else:
        return value


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior"""

    markevery: Optional[int] = None
    linestyle: str = "solid"
    plotevery: int = 1
    ncols: int = 4
    marker: Optional[str] = None
    color: Optional[str] = None
    label: Optional[str] = None
    x_metric: str = "epochs"
    y_metric: str = "loss"
    use_logscale: bool = True
    use_loglog: bool = True
    x_lim: Optional[Tuple[float, float]] = None
    y_lim: Optional[Tuple[float, float]] = None
    y_ticks: Optional[List[float]] = None


class PlotManager:
    """Manages plotting operations with consistent styling and configuration"""

    def __init__(self):
        self.plot_markers = ["o", "v", "^", "8", "s", "p", "P", "*", "h", "X", "D", "d"]
        self.plot_colors = ["#4285F4", "#5BC0EB", "#1F4E79", "brown", "grey"]
        self.additional_colors = ["#1f77b4", "#d62728", "#228B22", "#2ca02c", "#90EE90"]
        self.plot_colors_base = ["#1f77b4", "#5BC0EB", "brown", "#ff7700", "grey"]
        self.plot_colors_ub = ["#1f77b4", "#2ca02c", "#ff7700"]
        self.plot_colors_stochastic = ["#1f77b4", "#2ca02c", "brown", "#ff7700", "grey"]
        # Pink to brown gradient
        self.lipschitz_colors = ["#E91E63", "#9C27B0", "#673AB7", "#8D6E63", "#5D4037"]

        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Configure matplotlib settings"""
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["axes.spines.top"] = False
        plt.rcParams.update(
            {
                "axes.titlesize": 17,
                "axes.labelsize": 14,
                "legend.fontsize": 11,
                "lines.linewidth": 3,
                "lines.markersize": 8,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    def plot_assmpt_per_it(self, runs: List[Any], path: str, exp: int):
        """Plot assumption metrics per iteration"""
        plt.rcParams.update({"lines.linewidth": 3})

        # Define iterations to plot
        iterations = [0, 1, 5, 10, 50]

        # Get matching runs once at the beginning
        opt = {"name": "PoNoS", "forward_option": 10, "momentum": 2}

        matching_runs = self._get_matching_iteration_runs(runs, opt, exp)

        if not matching_runs:
            print("No matching runs found for optimizer:", opt)
            return

        # Create one figure per metric
        for metric_num, metric in enumerate(["orig_lip"], 1):
            plot_config = PlotConfig(
                markevery=1,
                x_metric="steps_along_g",
                ncols=5,
                use_loglog=True,
                x_lim=(0.001, 11),
            )
            plt.figure(metric_num, figsize=(6, 3))
            plot_config.y_metric = self._setup_ylabel(metric)

            all_x_data = []
            axvline_data = []  # Store axvline positions, colors, and markers

            # Plot each iteration
            for i, iteration in enumerate(iterations):
                plot_config.marker = self.plot_markers[i]
                plot_config.color = self.lipschitz_colors[i]

                # Extract data for this specific iteration from the matching runs
                for run in matching_runs:
                    opt_config = {
                        "name": "PoNoS",
                        "version": 10,
                        "iteration": iteration,
                    }
                    x_data, y_data, iteration = self._extract_per_it_data_single_run(
                        run, opt_config, metric
                    )
                    if x_data and len(y_data) > 0:
                        base_label = str(opt_config["iteration"] * 100)
                        plot_config.label = base_label
                        self._plot_single_line(x_data, y_data, plot_config)

                        # Remove specific values and add vertical line
                        exclude_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
                        x_data_clean = x_data.copy()
                        for val in exclude_values:
                            if val in x_data_clean:
                                x_data_clean.remove(
                                    val
                                )  # removes only first occurrence
                        if x_data_clean:
                            all_x_data.extend(x_data)
                            plt.axvline(
                                x=x_data_clean[0],
                                color=plot_config.color,
                                linestyle="--",
                                alpha=0.5,
                            )
                            # Store axvline info for marker placement
                            axvline_data.append(
                                {
                                    "x": x_data_clean[0],
                                    "color": plot_config.color,
                                    "marker": plot_config.marker,
                                }
                            )

                sharpness = run.plot_data["Eigenvalue 1"][iteration]
                plt.plot(
                    [0, 0.002],
                    [sharpness] * 2,
                    color=self.lipschitz_colors[i],
                    alpha=0.75,
                    linestyle="dashed",
                    label="_nolegend_",
                    linewidth=1,
                )
                x_data = []

            # Set axis limits
            if all_x_data:
                x_min, x_max = min(all_x_data), max(all_x_data)
                x_min = min(x_min, plot_config.x_lim[0])
                x_max = max(x_max, plot_config.x_lim[1])
                log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
                log_margin = (log_x_max - log_x_min) * 0.02
                x_min_with_margin = 10 ** (log_x_min - log_margin)
                x_max_with_margin = 10 ** (log_x_max + log_margin)
                plt.xlim(x_min_with_margin, x_max_with_margin)
            else:
                plt.xlim(plot_config.x_lim[0], plot_config.x_lim[1])

            y_min, y_max = plt.ylim()
            # Set y-ticks before other operations
            min_sharp = 2 / run.dataset.output_dim
            if y_min <= min_sharp:
                x_data_tmp = [0, plt.xlim()[1]]
                y_data_tmp = [min_sharp, min_sharp]
                plt.plot(
                    x_data_tmp,
                    y_data_tmp,
                    color="#FFD23F",
                    alpha=0.75,
                    linestyle="dashed",
                    label="_nolegend_",
                    linewidth=1,
                )
                found = False
                pos = -1
                for i, tick in enumerate(list(plt.yticks()[0])):
                    if np.abs(0.1 - tick) < 1e-06:
                        found = True
                        pos = i
                if found:
                    yticks_tmp = list(plt.yticks()[0])
                    del yticks_tmp[pos]
                    yticks_l_tmp = list(plt.yticks()[1])
                    del yticks_l_tmp[pos]
                    plt.yticks(yticks_tmp, yticks_l_tmp)
                plt.yticks(
                    list(plt.yticks()[0]) + [min_sharp], list(plt.yticks()[1]) + ["2/K"]
                )

            # Set labels and scales
            plt.xlabel(plot_config.x_metric)
            plt.ylabel(plot_config.y_metric)

            # Distribute markers
            if len(axvline_data) > 0:
                log_y_positions = np.linspace(
                    np.log10(y_min), np.log10(y_max), len(axvline_data)
                )
                y_positions = 10**log_y_positions
                for i, axv_info in enumerate(axvline_data):
                    plt.scatter(
                        axv_info["x"],
                        y_positions[i],
                        marker=axv_info["marker"],
                        color=axv_info["color"],
                        s=64,
                        zorder=5,
                        alpha=0.4,
                    )

            if matching_runs:
                plt.title(self._setup_title(matching_runs[0]))

            plt.legend(
                bbox_to_anchor=(0.5, -0.35),
                loc="lower center",
                borderaxespad=0,
                ncol=plot_config.ncols,
            )

            # Save plot for this metric
            if matching_runs:
                self._save_plot(
                    matching_runs[0],
                    path,
                    self._setup_ylabel(metric).replace(" ", "") + "_per_it",
                )

        plt.close("all")

    def plot_per_it(
        self,
        runs: List[Any],
        path: str,
        metrics: List[str],
        exp: int = 1,
        loss_type: str = "mse",
        c: float = 1e-4,
    ):
        """Plot per-iteration metrics"""
        mpl.rcParams["path.simplify"] = True
        mpl.rcParams["path.simplify_threshold"] = 1.0
        plt.rcParams.update({"lines.linewidth": 1.5})
        use_logscale = True

        if exp == 0:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 0,
                },
                {
                    "name": "PoNoS",
                    "forward_option": 0,
                    "momentum": 0,
                },
                {
                    "name": "SLS",
                    "forward_option": 10,
                    "momentum": 0,
                },
                {
                    "name": "CDAT",
                    "forward_option": 0,
                    "momentum": 0,
                },
                {
                    "name": "SAM",
                    "forward_option": 0,
                    "momentum": 0.9,
                },
            ]
        elif exp == 1:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 1,
                },
            ]
        elif exp == 2:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 0,
                },
                {
                    "name": "PoNoS",
                    "forward_option": 11,
                    "momentum": 0,
                },
                {
                    "name": "CDAT",
                    "forward_option": 0,
                    "momentum": 0,
                },
            ]
        elif exp == 3:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 2,
                },
            ]
        elif exp == 4:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 0,
                },
                {
                    "name": "PoNoS",
                    "forward_option": 11,
                    "momentum": 0,
                },
                {
                    "name": "SLS",
                    "forward_option": 10,
                    "momentum": 0,
                },
                {
                    "name": "CDAT",
                    "forward_option": 0,
                    "momentum": 0,
                },
                {
                    "name": "SAM",
                    "forward_option": 0,
                    "momentum": 0.9,
                },
            ]
        elif exp == 5:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 0,
                    "delta": 0.5,
                },
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 0,
                    "delta": 0.9,
                },
            ]
        elif exp == 6:
            optimizers = [
                {
                    "name": "warmup_GD",
                },
                {
                    "name": "warmup_GD_small",
                },
            ]
        elif exp == 7:
            optimizers = [
                {
                    "name": "PoNoS",
                    "forward_option": 10,
                    "momentum": 3,
                }
            ]
            use_logscale = False

        # Create one figure per metric
        for metric_num, metric in enumerate(metrics, 1):
            plot_config = PlotConfig(
                markevery=None,
                plotevery=1,
                ncols=5,
                marker=None,
                use_loglog=False,
                use_logscale=use_logscale,
            )
            plt.figure(metric_num, figsize=(6, 3))
            if runs[0].batch_size == 256:
                plot_config.x_metric = "Epoch"
            else:
                plot_config.x_metric = "Iteration"
            plot_config.y_metric = self._setup_ylabel(metric)

            # Special y-ticks for sharpness metrics
            if metric == "Sharpness x step":
                if c == 0.5:
                    plot_config.y_ticks = [0.1, 1 - c, 1, 2, 10]
                    plot_config.y_ticks_names = ["0.1", "1-c", "1", "2", "10"]
                else:
                    plot_config.y_ticks = [0.1, 1 - c, 2, 10]
                    plot_config.y_ticks_names = ["0.1", "1-c", "2", "10"]
                plot_config.y_lim = [0.05, 10]  # Note: the second value is not used

            all_x_data = []
            all_y_data = []
            enter = True
            gd_count = 0
            no_valid_runs = True

            for i, opt in enumerate(optimizers):
                # Get all runs for this optimizer
                optimizer_runs = self._get_matching_iteration_runs(
                    runs, opt, exp, loss_type
                )

                if len(optimizer_runs) == 0:
                    print("No runs found for optimizer:", opt)
                    continue

                no_valid_runs = False

                if len(optimizer_runs) > 1 and opt["name"] != "constant_stepsize_GD":
                    if metric_num == 1:
                        print("There is more than 1 optimizer:")
                        for run in optimizer_runs:
                            print(run.run_id, run.optimizer, run.epochs)
                    optimizer_runs = [optimizer_runs[0]]

                for run_idx, run in enumerate(optimizer_runs):
                    if metric_num == 1:
                        print("The filtered optimizers are:")
                        print(run.run_id, run.optimizer)
                    x_data, y_data = self._process_metric_data(run, metric)

                    # Subsample data for certain metrics and experiments
                    subsample_metrics = {
                        "Training Loss",
                        "Training Accuracy",
                        "Test Loss",
                        "Test Accuracy",
                        "Step Size",
                        "Gradient Norm",
                    }
                    subsample_exps = {0, 2, 5, 7}
                    if metric in subsample_metrics and exp in subsample_exps:
                        print(
                            "Subsampling data for metric and experiment:", metric, exp
                        )
                        x_data, y_data = self._subsample_data(x_data, y_data, step=10)

                    if x_data and len(y_data) > 0:
                        if (
                            len(optimizer_runs) > 1
                            and run.optimizer.opt_name == "constant_stepsize_GD"
                        ):
                            plot_config.color = self._get_color_for_optimizer(
                                opt, i + gd_count, exp
                            )
                            gd_count += 1
                        else:
                            plot_config.color = self._get_color_for_optimizer(
                                opt, i, exp
                            )
                        base_label = self._setup_labels(run, exp)
                        plot_config.label = base_label

                        all_x_data.extend(x_data)
                        all_y_data.extend(y_data)

                        if metric == "Sharpness x step" and enter:
                            self._add_reference_lines(x_data, y_data, c)
                            enter = False

                        if metric == "Avg Hidden Grad Norm":
                            plot_config.label = "Avg Hidden"
                            self._plot_single_line(x_data, y_data, plot_config)

                            x_upper, y_upper = self._process_metric_data(
                                run, "Max Hidden Grad Norm"
                            )
                            x_lower, y_lower = self._process_metric_data(
                                run, "Min Hidden Grad Norm"
                            )
                            plt.fill_between(
                                x_upper,
                                y_lower,
                                y_upper,
                                color=plot_config.color,
                                alpha=0.3,
                                label="_nolegend_",
                            )
                            all_x_data.extend(x_upper)
                            all_x_data.extend(x_lower)
                            all_y_data.extend(y_upper)
                            all_y_data.extend(y_lower)

                            x_data, y_data = self._process_metric_data(
                                run, "Bias Grad Norm"
                            )
                            all_x_data.extend(x_data)
                            all_y_data.extend(y_data)
                            plot_config.color = self._get_color_for_optimizer(
                                opt, i + 1, exp
                            )
                            plot_config.label = "Last Bias"
                            self._plot_single_line(x_data, y_data, plot_config)
                        else:
                            self._plot_single_line(x_data, y_data, plot_config)

            if no_valid_runs:
                print(
                    "No valid runs found for any optimizer. Skipping plot for metric:",
                    metric,
                )
                plt.close()
                continue

            # Set axis limits
            if all_x_data:
                x_min, x_max = min(all_x_data), max(all_x_data)
                x_margin = (x_max - x_min) * 0.01
                plt.xlim(x_min - x_margin, x_max + x_margin)

            # Set labels and scales
            plt.xlabel(plot_config.x_metric)
            plt.ylabel(plot_config.y_metric)

            # Set y-lim and y-ticks after other operations
            if (
                metric == "Eigenvalue 1"
                or metric == "Max Batch Eigenvalue 1"
                or metric == "Min Batch Eigenvalue 1"
                or metric == "Avg Batch Eigenvalue 1"
            ):
                if loss_type == "mse" and (
                    exp == 0 or exp == 2 or exp == 4 or exp == 5 or exp == 6
                ):
                    plt.yticks(
                        list(plt.yticks()[0])
                        + [2 / optimizer_runs[0].dataset.output_dim],
                        list(plt.yticks()[1]) + ["2/K"],
                    )
                    plt.axhline(
                        y=(2 / optimizer_runs[0].dataset.output_dim),
                        color="#FFD23F",
                        linestyle="dashed",
                        linewidth=1,
                        alpha=0.75,
                    )
                    if exp == 4:
                        plt.ylim(bottom=1e-2)
                    if exp == 0 and optimizer_runs[0].dataset.output_dim == 10:
                        plt.ylim(bottom=1e-1)
                    elif exp == 0 and optimizer_runs[0].dataset.output_dim == 100:
                        plt.ylim(bottom=1e-2)
                    elif exp == 0 and optimizer_runs[0].dataset.output_dim == 26:
                        plt.ylim(bottom=1e-2)

            if metric == "Sharpness x step":
                lower_lim = max(min(all_y_data), plot_config.y_lim[0])
                lower_lim = min(lower_lim, 1 - c - 0.2)
                plt.ylim(lower_lim, max(all_y_data))
            if plot_config.y_ticks:
                if lower_lim == 0.05:
                    plt.yticks(plot_config.y_ticks, plot_config.y_ticks_names)
                else:
                    plt.yticks(plot_config.y_ticks[1:], plot_config.y_ticks_names[1:])
            if (
                metric == "Step Size"
                and min(all_y_data, key=noneIsInfinite) == 0.000001
            ):
                plt.yticks([0.000001, 0.0001, 0.01, 1, 100, 1000])
            if metric == "Training Loss":
                plt.ylim(None, min(100, max(all_y_data)))
            if runs:
                plt.title(self._setup_title(runs[0]))

            plt.legend(
                bbox_to_anchor=(0.5, -0.35),
                loc="lower center",
                borderaxespad=0,
                ncol=plot_config.ncols,
            )

            # Save plot for this metric
            if isinstance(run.loss_fn, nn.CrossEntropyLoss):
                filename = f"{metric.replace(' ', '')}_ce_it_exp{exp}"
            else:
                filename = f"{metric.replace(' ', '')}_it_exp{exp}"
            if runs:
                self._save_plot(
                    runs[0],
                    path,
                    filename,
                )

        plt.close("all")

    def plot_eigenvalues(
        self, runs: List[Any], path: str, metrics: List[str], c: float
    ):
        """Plot per-iteration metrics"""
        c = 0.0001
        plt.rcParams.update({"lines.linewidth": 1.5})

        # Create one figure per metric
        for metric_num, metric in enumerate(
            metrics + ["Eigenvalues"], 1  # "PerturbedTrace", "Eigenvalues x step"], 1
        ):

            plot_config = PlotConfig(
                markevery=None, plotevery=1, ncols=5, marker=None, use_loglog=False
            )
            plt.figure(metric_num, figsize=(6, 3))
            plot_config.x_metric = "Iteration"
            plot_config.y_metric = self._setup_ylabel(metric)

            # Special y-ticks for sharpness metrics
            if metric == "Sharpness x step" or metric == "Eigenvalues x step":
                plot_config.y_ticks = [0.1, 1 - c, 2, 10]
                plot_config.y_ticks_names = ["0.1", "1-c", "2", "10"]
                plot_config.y_lim = [0.05, 10]  # Note: the second value is not used

            all_x_data = []
            all_y_data = []
            enter = True
            optimizer_runs = self._get_debugging_runs(runs)

            if len(optimizer_runs) == 0:
                print("No debugging runs found")
                continue

            for run_idx, run in enumerate(optimizer_runs):
                x_data, y_data_dict = self._extract_debugging_data(run, metric, enter)
                for i, y_data in enumerate(y_data_dict.values()):
                    if x_data and len(y_data) > 0:
                        opt = {"name": "PoNoS", "forward_option": 10}
                        plot_config.color = self._get_color_for_optimizer(opt, 0, 1)
                        plot_config.label = "trace"
                        if metric == "Eigenvalues x step":
                            plot_config.color = self._get_color_for_optimizer(opt, i, 6)
                            plot_config.label = "Eig " + str(i)

                        all_x_data.extend(x_data)
                        all_y_data.extend(y_data)

                        if (
                            metric == "Sharpness x step"
                            or metric == "Eigenvalues x step"
                        ) and enter:
                            self._add_reference_lines(x_data, y_data, c)
                            enter = False
                        if metric == "Eigenvalues":
                            self._add_reference_lines(
                                x_data,
                                y_data,
                                c=None,
                                g_flat=2 / run.dataset.output_dim,
                            )
                            enter = False
                        if metric != "Eigenvalues" or i == run.plot_metrics.num_eigs:
                            self._plot_single_line(x_data, y_data, plot_config)
                        else:
                            self._scatter_eigenvalues(x_data, y_data, plot_config)

            if metric == "Eigenvalues":
                plt.yticks(
                    list(plt.yticks()[0]) + [2 / optimizer_runs[0].dataset.output_dim],
                    list(plt.yticks()[1]) + ["2/K"],
                )
                plt.autoscale(enable=True, axis="y", tight=False)

            # Set axis limits
            if all_x_data:
                x_min, x_max = min(all_x_data), max(all_x_data)
                x_margin = (x_max - x_min) * 0.01
                plt.xlim(x_min - x_margin, x_max + x_margin)

            # Set labels and scales
            plt.xlabel(plot_config.x_metric)
            plt.ylabel(plot_config.y_metric)

            # Set y-lim and y-ticks after other operations
            if metric == "Sharpness x step" or metric == "Eigenvalues x step":
                lower_lim = max(min(all_y_data), plot_config.y_lim[0])
                lower_lim = min(lower_lim, 1 - c - 0.2)
                plt.ylim(lower_lim, max(all_y_data))
            if plot_config.y_ticks:
                if lower_lim == 0.05:
                    plt.yticks(plot_config.y_ticks, plot_config.y_ticks_names)
                else:
                    plt.yticks(plot_config.y_ticks[1:], plot_config.y_ticks_names[1:])
            if (
                metric == "Step Size"
                and min(all_y_data, key=noneIsInfinite) == 0.000001
            ):
                plt.yticks([0.000001, 0.0001, 0.01, 1, 100, 1000])
            if metric == "Training Loss":
                plt.ylim(None, min(100, max(all_y_data)))

            if runs:
                plt.title(self._setup_title(runs[0]))

            plt.legend(
                bbox_to_anchor=(0.5, -0.3),
                loc="lower center",
                borderaxespad=0,
                ncol=plot_config.ncols,
            )
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # keeps only one handle per label
            plt.legend(by_label.values(), by_label.keys())

            # Save plot for this metric
            if runs:
                self._save_plot(
                    runs[0],
                    path,
                    f"{metric.replace(' ', '')}_{self._float_to_filename(c)}",
                )
        plt.close("all")

    def _scatter_eigenvalues(self, x_data, y_data, plot_config):
        for x, y in zip(x_data, y_data):
            if y >= 0:
                plt.scatter(
                    x, y, marker=".", color="blue", s=15, label="pos-eig"
                )  # zorder=5, alpha=0.4)
            else:
                plt.scatter(
                    x,
                    -y,
                    marker="o",
                    color="red",
                    s=15,
                    facecolors="none",
                    label="neg-eig",
                )  # zorder=5, alpha=0.4)

    def _plot_single_line(
        self, x_data: List[float], y_data: List[float], config: PlotConfig
    ):
        """Plot a single line with the given configuration"""
        if config.use_logscale:
            if config.use_loglog:
                plt.loglog(
                    x_data,
                    y_data,
                    markevery=config.markevery,
                    label=config.label,
                    marker=config.marker,
                    linestyle=config.linestyle,
                    color=config.color,
                )
            else:
                plt.semilogy(
                    x_data,
                    y_data,
                    markevery=config.markevery,
                    label=config.label,
                    marker=config.marker,
                    linestyle=config.linestyle,
                    color=config.color,
                )
        else:
            plt.plot(
                x_data,
                y_data,
                markevery=config.markevery,
                label=config.label,
                marker=config.marker,
                linestyle=config.linestyle,
                color=config.color,
            )

    def _get_matching_runs(self, runs: List[Any], opt: Dict[str, Any]) -> List[Any]:
        """Get all runs that match the optimizer criteria"""
        matching_runs = []
        for run in runs:
            if self._matches_optimizer(run, opt) and self._is_valid_run(run, opt):
                matching_runs.append(run)
        return matching_runs

    def _get_matching_iteration_runs(
        self, runs: List[Any], opt: Dict[str, Any], exp: int, loss_type: str = "mse"
    ) -> List[Any]:
        """Get all runs that match the iteration criteria"""
        matching_runs = []
        best_step_size = {
            ("SAM", "CIFAR10", "CNN"): 0.1,
            ("SAM", "CIFAR10", "MLP"): 0.1,
            ("SAM", "CIFAR10", "resnet34"): 0.01,
            ("SAM", "CIFAR10", "vgg11"): 0.1,
            ("SAM", "CIFAR10", "tinyVIT"): 0.01,
            ("SAM", "CIFAR10", "densenet121"): 0.1,
            ("SAM", "CIFAR10", "wide_resnet50_2"): 0.0001,
            ("SAM", "CIFAR100", "CNN"): 0.01,
            ("SAM", "CIFAR100", "MLP"): 0.1,
            ("SAM", "CIFAR100", "resnet34"): 0.01,
            ("SAM", "CIFAR100", "vgg11"): 0.01,
            ("SAM", "CIFAR100", "wide_resnet50_2"): 0.001,
            ("SAM", "CIFAR100", "densenet121"): 0.1,
            ("SAM", "CIFAR100", "tinyVIT"): 0.01,
            ("SAM", "SVHN", "CNN"): 0.1,
            ("SAM", "SVHN", "MLP"): 0.1,
            ("SAM", "SVHN", "resnet34"): 0.001,
            ("SAM", "SVHN", "vgg11"): 0.1,
            ("SAM", "SVHN", "tinyVIT"): 0.01,
            ("SAM", "SVHN", "densenet121"): 0.1,
            ("SAM", "SVHN", "wide_resnet50_2"): 0.0001,
            ("SAM", "EMNIST", "CNN"): 0.01,
            ("SAM", "EMNIST", "MLP"): 0.1,
            ("SAM", "EMNIST", "resnet34"): 0.001,
            ("SAM", "EMNIST", "vgg11"): 0.1,
            ("SAM", "EMNIST", "tinyVIT"): 0.01,
            ("SAM", "EMNIST", "densenet121"): 0.1,
            ("SAM", "EMNIST", "wide_resnet50_2"): 0.0001,
        }
        if loss_type == "mse":
            loss_fn = nn.MSELoss
        elif loss_type == "ce":
            loss_fn = nn.CrossEntropyLoss

        for run in runs:
            if exp == 0:
                if (
                    opt["name"] != "SAM"
                    and run.optimizer.opt_name == opt["name"]
                    and run.optimizer.forward_option == opt["forward_option"]
                    and run.optimizer.momentum == opt["momentum"]
                    and isinstance(run.loss_fn, loss_fn)
                ):
                    matching_runs.append(run)
                    print(run.run_id, run.optimizer)

                if (
                    opt["name"] == "SAM"
                    and run.optimizer.opt_name == opt["name"]
                    and run.optimizer.step_size
                    == best_step_size.get(
                        ("SAM", run.dataset.name, run.model.model_type)
                    )
                    and run.optimizer.momentum == opt["momentum"]
                    and isinstance(run.loss_fn, loss_fn)
                ):
                    matching_runs.append(run)
                    print(run.run_id, run.optimizer)

            elif exp == 1 or exp == 2 or exp == 3 or exp == 4:
                if (
                    run.optimizer.opt_name == opt["name"]
                    and run.optimizer.forward_option == opt["forward_option"]
                    and run.optimizer.momentum == opt["momentum"]
                    and isinstance(run.loss_fn, loss_fn)
                ):
                    matching_runs.append(run)
                    print(run.run_id, run.optimizer)

            elif exp == 5:
                if (
                    run.optimizer.opt_name == opt["name"]
                    and run.optimizer.forward_option == opt["forward_option"]
                    and run.optimizer.momentum == opt["momentum"]
                    and run.optimizer.delta == opt["delta"]
                    and isinstance(run.loss_fn, loss_fn)
                ):
                    matching_runs.append(run)
                    print(run.run_id, run.optimizer)

            elif exp == 6:
                if run.optimizer.opt_name == opt["name"] and isinstance(
                    run.loss_fn, loss_fn
                ):
                    matching_runs.append(run)
                    print(run.run_id, run.optimizer)

            elif exp == 7:
                if (
                    run.optimizer.opt_name == opt["name"]
                    and run.optimizer.forward_option == opt["forward_option"]
                    and run.optimizer.momentum == opt["momentum"]
                    and run.use_bias == False
                    and isinstance(run.loss_fn, loss_fn)
                ):
                    matching_runs.append(run)
                    print(run.run_id, run.optimizer)

        return matching_runs

    def _get_debugging_runs(self, runs: List[Any]):
        matching_runs = []
        for run in runs:
            if "debugging" in run.plot_metrics.metrics:
                matching_runs.append(run)
        return matching_runs

    def _extract_per_it_data_single_run(
        self, run: Any, opt: Dict[str, Any], metric: str
    ) -> Tuple[List[float], List[float]]:
        """Extract data for a single run in per-iteration plotting"""
        #        iteration = min(opt["iteration"] * 100, len(run.plot_data[metric]) - 1)
        iteration = min(opt["iteration"], len(run.plot_data[metric]) - 1)
        data_list = run.plot_data[metric][iteration]

        if len(data_list) > 0:
            y_data = data_list
            x_data = copy.deepcopy(run.plot_data["g_steps"][iteration])
            return x_data, y_data, iteration

        return [], [], iteration

    def _extract_debugging_data(
        self, run: Any, metric: str, enter: bool
    ) -> Tuple[List[float], List[float]]:
        """Extract data for a single run in iteration plotting"""
        if metric == "Eigenvalues":
            y_data_dict = dict.fromkeys(range(run.plot_metrics.num_eigs), [])
            # x_data = list(range(len(run.plot_data["Eigenvalue 1"])))
            x_data = list(
                np.arange(len(run.plot_data["Eigenvalue 1"]))
                * run.plot_metrics.sharpness_every
            )
            for i in range(run.plot_metrics.num_eigs):
                y_data_dict[i] = copy.deepcopy(
                    run.plot_data["Eigenvalue " + str(i + 1)]
                )
                # print("Eigenvalue ", i+1, " with length ", len(y_data_dict[i]), " data: ", y_data_dict[i][:5], )
            _, y_data = self._process_metric_data(run, "Trace")
            y_data_dict[run.plot_metrics.num_eigs] = y_data
        elif metric == "Eigenvalues x step":
            n_eigs = min(run.plot_metrics.num_eigs, 5)
            y_data_dict = dict.fromkeys(range(run.plot_metrics.num_eigs), [])
            x_data = list(range(len(run.plot_data["Eigenvalue 1"])))

            arr = np.array(
                [run.plot_data["Eigenvalue " + str(i + 1)] for i in range(n_eigs)]
            )
            sorted_arr = np.sort(arr, axis=0)[::-1].tolist()

            for i in range(n_eigs):
                y_data_dict[i] = (
                    np.array(sorted_arr[i]) * np.array(run.plot_data["Step Size"])
                ).tolist()

        elif metric == "PerturbedTrace":
            trace = run.plot_data["Trace"]
            perturbed_trace = run.plot_data[metric]
            x_data = list(range(len(perturbed_trace)))
            if len(perturbed_trace) == len(trace):
                y_data_dict = {
                    0: np.abs(np.array(trace) - np.array(perturbed_trace)).tolist()
                }
            else:
                y_data_dict = {0: perturbed_trace}
                print(
                    f"Something went wrong in the (perturbed) trace extraction, len(perturbed_trace) != len(trace), {len(perturbed_trace)} != {len(trace)}"
                )
        else:
            x_data, y_data = self._process_metric_data(run, metric)
            y_data_dict = {0: y_data}

        if hasattr(run.plot_metrics, "after_it") and run.plot_metrics.after_it != 0:
            return (
                np.array(x_data) + np.ones(len(x_data)) * run.plot_metrics.after_it
            ).tolist(), y_data_dict
        else:
            return x_data, y_data_dict

    def _add_reference_lines(
        self, x_data: List[float], y_data: List[float], c=None, g_flat=None
    ):
        """Add reference lines for sharpness x step plots"""
        if c:
            plt.rcParams.update({"lines.linewidth": 3})
        else:
            plt.rcParams.update({"lines.linewidth": 1})

        # Add reference line at y=2
        x_data_tmp = [0, len(y_data) * 100]
        y_data_tmp = [2, 2]
        plt.plot(
            x_data_tmp,
            y_data_tmp,
            color="#8B008B",
            alpha=0.75,
            linestyle="dashed",
            label="_nolegend_",
        )
        if c is not None:
            # Add reference line at y=0.5
            y_data_tmp = [1 - c, 1 - c]
            plt.plot(
                x_data_tmp,
                y_data_tmp,
                color="black",
                alpha=0.75,
                linestyle="dashed",
                label="_nolegend_",
            )
        elif g_flat is not None:
            y_data_tmp = [g_flat, g_flat]
            plt.plot(
                x_data_tmp,
                y_data_tmp,
                color="#D4AF37",
                alpha=0.5,
                linestyle="dashed",
                label="_nolegend_",
            )

        plt.rcParams.update({"lines.linewidth": 1.5})

    def _save_plot(self, run: Any, path: str, filename: str):
        """Save the current plot to file"""
        plot_dir = os.path.join(
            path, "plots", run.dataset.name, run.model.model_type, str(run.batch_size)
        )
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename + ".pdf"), bbox_inches="tight")

    def _calculate_min_indices(
        self, runs: List[Any], metrics: List[str]
    ) -> Dict[str, int]:
        """Calculate minimum indices for training loss"""
        min_indices = {}
        if "Training Loss Min" in metrics:
            for run in runs:
                if hasattr(run, "plot_data") and "Training Loss" in run.plot_data:
                    try:
                        min_index = np.nanargmin(
                            np.array(run.plot_data["Training Loss"])
                        )
                    except ValueError:
                        min_index = -1
                    min_indices[run.run_id] = min_index
        return min_indices

    def _extract_offline_data(
        self,
        runs: List[Any],
        opt: Dict[str, Any],
        metric: str,
        min_indices: Dict[str, int],
    ) -> Tuple[List[float], List[float], Any]:
        """Extract data for offline plotting"""
        x_data, y_data = [], []
        run_example = None

        for run in runs:
            if not self._matches_optimizer(run, opt):
                continue

            if self._is_valid_run(run, opt):
                x_data.append(self._get_x_value(run, opt))
                y_data.append(self._get_y_value(run, metric, min_indices))
                run_example = run

        return x_data, y_data, run_example

    def _extract_per_it_data(
        self, runs: List[Any], opt: Dict[str, Any], metric: str
    ) -> Tuple[List[float], List[float], Any]:
        """Extract data for per-iteration plotting"""
        for run in runs:
            if (
                run.optimizer.opt_name == opt["name"]
                and run.optimizer.forward_option == opt["version"]
                and run.optimizer.momentum == 0.9
            ):

                iteration = min(opt["iteration"] * 100, len(run.plot_data[metric]) - 1)
                data_list = run.plot_data[metric][iteration]

                if len(data_list) > 0:
                    y_data = data_list
                    x_data = copy.deepcopy(run.plot_data["g_steps"][iteration])
                    return x_data, y_data, run

        return [], [], None

    def _extract_iteration_data(
        self, runs: List[Any], opt: Dict[str, Any], metric: str, enter: bool
    ) -> Tuple[List[float], List[float], Any]:
        """Extract data for iteration plotting"""
        for run in runs:
            if (
                run.optimizer.opt_name == opt["name"]
                and run.optimizer.forward_option == opt["version"]
                and run.optimizer.c == opt["c"]
                and run.optimizer.momentum == opt["momentum"]
            ):

                x_data, y_data = self._process_metric_data(run, metric)
                return x_data, y_data, run

        return [], [], None

    def _process_metric_data(
        self, run: Any, metric: str
    ) -> Tuple[List[float], List[float]]:
        """Process metric data based on metric type"""
        every = run.plot_metrics.sharpness_every
        if metric == "Sharpness x step":
            if len(run.plot_data["Eigenvalue 1"]) != len(run.plot_data["Step Size"]):
                new_data = [
                    step
                    for i, step in enumerate(run.plot_data["Step Size"])
                    if i % every == 0
                ]
                if len(run.plot_data["Eigenvalue 1"]) == (len(new_data) + 1):
                    new_data.append(run.plot_data["Step Size"][-1])
                    run.plot_data["Sharpness x step"] = np.array(
                        run.plot_data["Eigenvalue 1"]
                    ) * np.array(new_data)
                else:
                    run.plot_data["Sharpness x step"] = (
                        np.array(run.plot_data["Eigenvalue 1"]) * np.array(new_data)
                    )[:-1]
            else:
                run.plot_data["Sharpness x step"] = np.array(
                    run.plot_data["Eigenvalue 1"]
                ) * np.array(run.plot_data["Step Size"])

            x_data = list(range(0, (len(run.plot_data[metric]) - 1) * every, every))
            x_data.append(len(run.plot_data[metric]) * every - 1)
        elif (
            metric == "Eigenvalue 1"
            or metric == "Avg Batch Eigenvalue 1"
            or metric == "Min Batch Eigenvalue 1"
            or metric == "Max Batch Eigenvalue 1"
        ):
            x_data = list(np.arange(len(run.plot_data[metric])) * every)
        elif metric == "Backtracks" or metric == "Function Evaluations":
            run.plot_data[metric] = np.convolve(
                run.plot_data[metric], np.ones(25) / 25, mode="valid"
            )
            x_data = list(range(len(run.plot_data[metric])))
        else:
            x_data = list(range(len(run.plot_data[metric])))

        if hasattr(run.plot_metrics, "after_it") and run.plot_metrics.after_it != 0:
            if (
                metric == "Eigenvalue 1"
                or metric == "Avg Hidden Grad Norm"
                or metric == "Zero Grad Entries"
                or metric == "Zero Activations"
                or metric == "Max Hidden Grad Norm"
                or metric == "Min Hidden Grad Norm"
                or metric == "Bias Grad Norm"
            ):
                return (
                    np.array(x_data) + np.ones(len(x_data)) * run.plot_metrics.after_it
                ).tolist(), run.plot_data[metric]
            elif metric == "Trace":
                return x_data, run.plot_data[metric]
            else:
                return (
                    x_data[run.plot_metrics.after_it :],
                    run.plot_data[metric][run.plot_metrics.after_it :],
                )
        else:
            return x_data, run.plot_data[metric]

    def _subsample_data(
        self, x_data: List[float], y_data: List[float], step: int = 10
    ) -> Tuple[List[float], List[float]]:

        x_subsampled = x_data[::step]
        y_subsampled = y_data[::step]

        return x_subsampled, y_subsampled

    def _matches_optimizer(self, run: Any, opt: Dict[str, Any]) -> bool:
        """Check if run matches optimizer specification"""
        return run.optimizer.opt_name == opt["name"]

    def _is_valid_run(self, run: Any, opt: Dict[str, Any]) -> bool:
        """Check if run is valid for plotting"""
        if run.optimizer.opt_name == "PoNoS":
            return (
                run.optimizer.forward_option == opt["version"]
                and run.optimizer.c not in [0.05, 0.3]
                and run.optimizer.momentum != 0.9
            )
        elif run.optimizer.opt_name == "CDAT":
            return run.optimizer.c != 2.50
        elif run.optimizer.opt_name == "SAM":
            return run.optimizer.momentum == opt["version"]
        return True

    def _get_x_value(self, run: Any, opt: Dict[str, Any]) -> float:
        """Get x-axis value for the run"""
        if run.optimizer.opt_name == "SAM":
            return run.optimizer.step_size
        return run.optimizer.c

    def _get_y_value(self, run: Any, metric: str, min_indices: Dict[str, int]) -> float:
        """Get y-axis value for the run"""
        if metric == "Training Loss Min":
            m_name = metric.replace(" Min", "")
            data = run.plot_data[m_name][min_indices[run.run_id]]
            return 1.0 if np.isnan(data) or np.isinf(data) else data
        else:  # if "Training Loss Min" in min_indices:
            return self._get_indexed_value(run, metric, min_indices[run.run_id])
        # TODO: this would have been when we simply take the values from the last iterates
        return run.plot_data[metric][-1]

    def _get_indexed_value(self, run: Any, metric: str, min_index: int) -> float:
        """Get value at specific index with proper handling"""

        def handle_neg_sharpness(
            run: Any, metric: str, min_index: int, data: float
        ) -> float:
            j = 1
            while data < 0:
                index = int((min_index + 5 * j) / 5)
                try:
                    data = run.plot_data[metric][index]
                except:
                    pass
                if j < 0:
                    j = -j + 1
                else:
                    j = -j
            if j != -1:
                warnings.warn(
                    "There is a negative average shaprness value and no valid replacement, using the next positive value instead"
                )
            return data

        if metric == "Eigenvalue 1":
            index = int(min_index / 100)
            data = run.plot_data[metric][index]
            if data < 0:
                data = handle_neg_sharpness(run, metric, min_index, data)
            return 1000.0 if np.isnan(data) or np.isinf(data) else data
        elif metric == "Average Batch Eigenvalue 1":
            index = int(min_index / 5)
            data = run.plot_data[metric][index]
            if data < 0:
                data = handle_neg_sharpness(run, metric, min_index, data)
            return 1000.0 if np.isnan(data) or np.isinf(data) else data
        elif metric == "Training Loss":
            data = run.plot_data[metric][-1]
            return 1.0 if np.isnan(data) or np.isinf(data) else data
        elif "Accuracy" in metric:
            data = run.plot_data[metric][min_index]
            return 0.01 if np.isnan(data) or np.isinf(data) else data

        return run.plot_data[metric][min_index]

    def _get_color_for_optimizer(
        self, opt: Dict[str, Any], index: int, exp: int = 1
    ) -> str:
        """Get color for optimizer"""
        if exp == 0 or exp == 1 or exp == 3 or exp == 7:
            return self.plot_colors_base[index]
        elif exp == 2:
            return self.plot_colors_ub[index]
        elif exp == 4:
            return self.plot_colors_stochastic[index]

    def _reorder_lists(
        self, values: List[float], companion_list: List[float], ascending: bool = True
    ) -> Tuple[List[float], List[float]]:
        """Reorder two lists based on values in the first list"""
        if len(values) != len(companion_list):
            raise ValueError("Both lists must have the same length")

        paired_data = list(zip(values, companion_list))
        sorted_pairs = sorted(paired_data, key=lambda x: x[0], reverse=not ascending)

        if not sorted_pairs:
            return [], []

        sorted_values, reordered_companion = zip(*sorted_pairs)
        return list(sorted_values), list(reordered_companion)

    def _float_to_filename(self, value: float) -> str:
        """Convert float to filename-safe string"""
        if value == 0:
            return "0"

        str_val = f"{value:.10f}".rstrip("0").rstrip(".")
        filename_str = str_val.replace(".", "_")

        if filename_str.startswith("0_"):
            filename_str = filename_str[2:]

        return filename_str

    def _setup_labels(self, run: Any, exp: int = 0) -> str:
        """Setup labels for plots"""
        if run.plot_metrics.label != "Optimizer":
            raise ValueError("Not a valid label for plot")

        opt_name = run.optimizer.opt_name
        forward_option = run.optimizer.forward_option
        delta = run.optimizer.delta

        label_mapping = {
            ("SLS", 10): "LS",
            ("PoNoS", 0): "PoNLS",
            ("PoNoS", 10): "NLS",
            ("PoNoS", 11): "NLS-ub",
            ("CDAT", None): "CDAT",
            ("SAM", None): "SAM",
        }

        label = label_mapping.get(
            (opt_name, forward_option), label_mapping.get((opt_name, None), opt_name)
        )

        # Special handling for warmup_GD
        if opt_name == "warmup_GD":
            label = "GD-warmup"
        if opt_name == "warmup_GD_small":
            label = "GD-warmup-ub"

        # Special handling for PoNoS with different delta values for delta ablation
        if opt_name == "PoNoS" and forward_option == 10 and exp == 5:
            label = f"NLS-δ={delta}"

        return label

    def _setup_ylabel(self, ymetric: str) -> str:
        """Setup y-axis labels"""
        ylabel_mapping = {
            "Step Size": "Step Size",
            "lipschitz": "Hess_lip",
            "orig_lip": "lipschitz",
            "Training Loss Min": "Training Loss",
            "Eigenvalue 1": "Sharpness",
            "Avg Batch Eigenvalue 1": "Avg Batch Sharpness",
            "Min Batch Eigenvalue 1": "Min Batch Sharpness",
            "Max Batch Eigenvalue 1": "Max Batch Sharpness",
            "Average Batch Eigenvalue 1": "Average Sharpness",
            "Sharpness x step": "Sharpness x Step Size",
            "Zero Activations": "Zero Activations (%)",
            "Training Accuracy": "Train Accuracy (%)",
            "Test Accuracy": "Test Accuracy (%)",
            "Zero Grad Entries": "Zero-Entries of the Gradient (%)",
            "Avg Hidden Grad Norm": "Layer-wise Grad Norm",
        }
        return ylabel_mapping.get(ymetric, ymetric)

    def _setup_title(
        self, run: Any, include_model: bool = True, include_dataset: bool = True
    ) -> str:
        """Setup plot title"""
        title_parts = []

        if include_dataset:
            dataset_name = run.dataset.name
            title_parts.append(dataset_name)

        if include_model:
            model_type = run.model.model_type
            title_parts.append(model_type)

        return " - ".join(title_parts)


# Global instance for backward compatibility
_plot_manager = PlotManager()


def plot_assmpt_per_it(runs: List[Any], path: str, exp: int):
    """Plot assumption metrics per iteration"""
    _plot_manager.plot_assmpt_per_it(runs, path, exp)


def plot_per_it(
    runs: List[Any],
    path: str,
    metrics: List[str],
    exp: int = 1,
    loss_type: str = "mse",
    c: float = 1e-4,
):
    """Plot per-iteration metrics"""
    _plot_manager.plot_per_it(runs, path, metrics, exp, loss_type, c)


def plot_eigenvalues(runs: List[Any], path: str, metrics: List[str], c: float):
    """Plot offline metrics per iteration and eigenvalues"""
    _plot_manager.plot_eigenvalues(runs, path, metrics, c)
