import torch.optim as optim
import torch as torch
from eos_line_search.utils import *
from eos_line_search.optimizers import SLS, PoNoS, SAM, CDAT
from eos_line_search.experiment import *
from eos_line_search.run import *
from eos_line_search.plot import *

warmup_counter = 0


def setup_optimizer(run):
    if run.optimizer.opt_name == "GD":
        opt_obj = optim.SGD(
            run.model.model_obj.parameters(),
            lr=run.optimizer.step_size,
            momentum=run.optimizer.momentum,
        )
    elif (
        run.optimizer.opt_name == "warmup_GD"
        or run.optimizer.opt_name == "warmup_GD_small"
    ):
        opt_obj = optim.SGD(
            run.model.model_obj.parameters(), lr=run.optimizer.step_size
        )
    elif run.optimizer.opt_name == "CDAT":
        opt_obj = CDAT.CDAT(
            run.model.model_obj.parameters(),
            sigma=run.optimizer.c,
            eps=run.optimizer.eps,
        )
    elif run.optimizer.opt_name == "SLS":
        opt_obj = SLS.SLS(
            run.model.model_obj.parameters(),
            c=run.optimizer.c,
            init_step_size=run.optimizer.init_step_size,
            max_eta=run.optimizer.max_step_size,
            n_batches_per_epoch=run.num_batches,
            beta=run.optimizer.delta,
            reset_option=run.optimizer.reset_option,
            forward_option=run.optimizer.forward_option,
            eps=run.optimizer.eps,
        )
    elif run.optimizer.opt_name == "PoNoS":
        opt_obj = PoNoS.PoNoS(
            run.model.model_obj.parameters(),
            c=run.optimizer.c,
            init_step_size=run.optimizer.init_step_size,
            max_eta=run.optimizer.max_step_size,
            n_batches_per_epoch=run.num_batches,
            beta=run.optimizer.delta,
            reset_option=run.optimizer.reset_option,
            forward_option=run.optimizer.forward_option,
            eps=run.optimizer.eps,
            zhang_xi=run.optimizer.xi,
            num_classes=run.dataset.output_dim,
        )
    elif run.optimizer.opt_name == "SAM":
        opt_obj = SAM.SAMSGD(
            run.model.model_obj.parameters(),
            lr=run.optimizer.step_size,
            rho=run.optimizer.rho,
            momentum=run.optimizer.momentum,
            weight_decay=run.optimizer.weight_decay,
        )
    else:
        raise ValueError("Not a valid optimizer")

    return opt_obj


def opt_step(loss, run, iteration):
    along_g_dict = {}
    if run.optimizer.opt_name == "GD":
        run.opt_obj.step()

        # Record metrics
        final_step_size = run.optimizer.step_size
        init_step_size = run.optimizer.step_size
        backtracks = 0
        function_evaluations = 1
    elif (
        run.optimizer.opt_name == "warmup_GD"
        or run.optimizer.opt_name == "warmup_GD_small"
    ):
        classes = run.dataset.output_dim
        increase_factor = 1.25
        stepsize_post_warmup = classes * 0.9

        if run.optimizer.opt_name == "warmup_GD":
            global warmup_counter
            stepsize_threshold = 1.25  # 1.01
            num_steps_counter = 1  # 10

            if warmup_counter <= num_steps_counter:
                if (
                    run.optimizer.step_size * increase_factor
                    <= classes * stepsize_threshold
                ):
                    run.opt_obj.param_groups[0]["lr"] = (
                        run.optimizer.step_size * increase_factor
                    )
                elif (
                    run.optimizer.step_size * increase_factor
                    > classes * stepsize_threshold
                ):
                    warmup_counter += 1
                    run.opt_obj.param_groups[0]["lr"] = classes * stepsize_threshold
            else:
                run.opt_obj.param_groups[0]["lr"] = stepsize_post_warmup
            run.optimizer.step_size = run.opt_obj.param_groups[0]["lr"]
            run.opt_obj.step()

        elif run.optimizer.opt_name == "warmup_GD_small":
            stepsize_threshold = 0.9  # 1.01

            if (
                run.optimizer.step_size * increase_factor
                <= classes * stepsize_threshold
            ):
                run.opt_obj.param_groups[0]["lr"] = (
                    run.optimizer.step_size * increase_factor
                )
            else:
                run.opt_obj.param_groups[0]["lr"] = stepsize_post_warmup
            run.optimizer.step_size = run.opt_obj.param_groups[0]["lr"]
            run.opt_obj.step()

        # Record metrics
        final_step_size = run.optimizer.step_size
        init_step_size = run.optimizer.step_size
        backtracks = 0
        function_evaluations = 1
    elif run.optimizer.opt_name == "SAM":
        run.opt_obj.step(loss)

        # Record metrics
        final_step_size = run.optimizer.step_size
        init_step_size = run.optimizer.step_size
        backtracks = 0
        function_evaluations = 2
    elif run.optimizer.opt_name == "CDAT":
        step_size = run.opt_obj.step()

        # Record metrics
        final_step_size = step_size
        init_step_size = step_size
        backtracks = 0
        function_evaluations = 3
    elif run.optimizer.opt_name == "SLS" or run.optimizer.opt_name == "PoNoS":
        check_Lw_asmpt = "Lw_asmpt" in run.plot_metrics.metrics and (
            (iteration % 100 == 0 and iteration < 1000)
            or iteration % 1000 == 0
            or iteration == run.epochs - 1
        )
        (
            final_step_size,
            init_step_size,
            backtracks,
            function_evaluations,
            along_g_dict,
        ) = run.opt_obj.step(
            closure=loss,
            check_Lw_asmpt=check_Lw_asmpt,
        )
    else:
        raise ValueError("Not a valid optimizer")

    return (
        final_step_size,
        init_step_size,
        backtracks,
        function_evaluations,
        along_g_dict,
    )
