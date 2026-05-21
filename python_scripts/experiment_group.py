from eos_line_search.optim import Optim

optimizers = []

for step in [-1]:
    optimizers.append(
        Optim(
            opt_name="SAM", step_size=step, momentum=0.9, weight_decay=0.0001, rho=0.05
        )
    )

for sigma in [2.06]:
    optimizers.append(Optim(opt_name="CDAT", c=sigma))

for ct in [1e-4]:
    optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=11,
            eps=0,
            xi=1,
            momentum=0,
        )
    )
    optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=10,
            eps=0,
            xi=1,
            momentum=0,
        )
    )
    optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=1,
            forward_option=0,
            eps=0,
            momentum=0,
        )
    )
    optimizers.append(
        Optim(
            opt_name="SLS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=10,
            eps=0,
            momentum=0,
        )
    )

debug_optimizers = []
debug_optimizers.append(
    Optim(
        opt_name="PoNoS",
        init_step_size=1,
        max_step_size=10000,
        c=1e-4,
        delta=0.5,
        reset_option=0,
        forward_option=10,
        eps=0,
        momentum=1,
    )
)

NLS_ub_optimizers = []
for sigma in [2.06]:
    NLS_ub_optimizers.append(Optim(opt_name="CDAT", c=sigma))

NLS_ub_optimizers.append(
    Optim(
        opt_name="PoNoS",
        init_step_size=1,
        max_step_size=10000,
        c=ct,
        delta=0.5,
        reset_option=0,
        forward_option=11,
        eps=0,
        xi=1,
        momentum=0,
    )
)

NLS_ub_optimizers.append(
    Optim(
        opt_name="PoNoS",
        init_step_size=1,
        max_step_size=10000,
        c=ct,
        delta=0.5,
        reset_option=0,
        forward_option=10,
        eps=0,
        xi=1,
        momentum=0,
    )
)

assmpt_optimizers = [
    Optim(
        opt_name="PoNoS",
        init_step_size=1,
        max_step_size=10000,
        c=0.0001,
        delta=0.5,
        reset_option=0,
        forward_option=10,
        eps=0,
        momentum=2,
    )
]

stochastic_optimizers = []

for step in [-1]:
    stochastic_optimizers.append(
        Optim(
            opt_name="SAM", step_size=step, momentum=0.9, weight_decay=0.0001, rho=0.05
        )
    )

for sigma in [2.06]:
    stochastic_optimizers.append(Optim(opt_name="CDAT", c=sigma))

for ct in [1e-4]:
    stochastic_optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=11,
            eps=0,
            xi=1,
            momentum=0,
        )
    )
    stochastic_optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=10,
            eps=0,
            xi=1,
            momentum=0,
        )
    )
    stochastic_optimizers.append(
        Optim(
            opt_name="SLS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=10,
            eps=0,
            momentum=0,
        )
    )

delta_ablation_optimizers = []

for ct in [1e-4]:
    delta_ablation_optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.5,
            reset_option=0,
            forward_option=10,
            eps=0,
            xi=1,
            momentum=0,
        )
    )
    delta_ablation_optimizers.append(
        Optim(
            opt_name="PoNoS",
            init_step_size=1,
            max_step_size=10000,
            c=ct,
            delta=0.9,
            reset_option=0,
            forward_option=10,
            eps=0,
            xi=1,
            momentum=0,
        )
    )

warmup_optimizers = []
warmup_optimizers.append(Optim(opt_name="warmup_GD_small", step_size=0.01, momentum=0))
warmup_optimizers.append(Optim(opt_name="warmup_GD", step_size=0.01, momentum=0))

no_bias_optimizers = [
    Optim(
        opt_name="PoNoS",
        init_step_size=1,
        max_step_size=10000,
        c=1e-4,
        delta=0.5,
        reset_option=0,
        forward_option=10,
        eps=0,
        momentum=3,
    )
]

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
