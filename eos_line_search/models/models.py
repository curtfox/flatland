import torch
import torch.nn as nn
import torchvision
import math
from eos_line_search.experiment import *
from torchinfo import summary
import vit_pytorch as vit

from collections import OrderedDict


def adapt_resnet_for_cifar(model, keep_maxpool=True):
    """
    Convert an ImageNet-style torchvision ResNet to a CIFAR-style stem in-place,
    but keep an option to retain the initial maxpool to avoid excessive activation
    memory growth on large batches with small images.

    Behavior:
      - Replace conv1 (7x7, stride=2) with 3x3 conv.
        * stride is set to 1 so the receptive field is CIFAR-friendly.
      - If keep_maxpool=True (default), keep model.maxpool (if present).
        This yields intermediate spatial sizes like: 32 -> 32 (conv1) -> 16 (maxpool)
        -> layer1..layer4 producing final layer4 roughly 2x2 for 32x32 inputs.
      - If keep_maxpool=False, maxpool is removed (original "cifar stem"), giving
        larger HxW (e.g., layer4 ~ 4x4 for 32x32 input), which can increase memory.

    This function only modifies the stem and maxpool; it does not otherwise change
    layer channel counts or block structure.
    """
    # Replace conv1: 7x7 stride2 -> 3x3 stride=1 (CIFAR-style)
    if hasattr(model, "conv1"):
        in_ch = model.conv1.in_channels
        out_ch = model.conv1.out_channels
        bias_present = model.conv1.bias is not None
        new_conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias_present,
        )

        # Try to preserve some initialization by copying the centered 3x3 patch
        try:
            with torch.no_grad():
                old_w = model.conv1.weight.data  # [out, in, 7, 7]
                if old_w.shape[2] >= 3:
                    cz = old_w.shape[2] // 2
                    center3 = old_w[:, :, cz - 1 : cz + 2, cz - 1 : cz + 2].clone()
                    if center3.shape == new_conv1.weight.data.shape:
                        new_conv1.weight.data.copy_(center3)
        except Exception:
            # fallback: leave default initialization
            pass

        model.conv1 = new_conv1

    # Keep or remove the initial maxpool according to keep_maxpool
    if hasattr(model, "maxpool"):
        if keep_maxpool:
            # Ensure we have a maxpool (leave as-is)
            # but if it was removed previously (Identity), restore a standard maxpool
            if isinstance(model.maxpool, nn.Identity):
                model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # remove maxpool (CIFAR stem)
            model.maxpool = nn.Identity()

    return model


def print_spatial_shapes(model, input_size=(1, 3, 32, 32), device="cpu"):
    """
    Run a single forward and print shapes of outputs of layer1..layer4 to help debug spatial dims.
    """
    model.eval()
    x = torch.randn(input_size).to(device)
    shapes = {}
    hooks = []

    def make_hook(name):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                shapes[name] = tuple(out.shape)

        return hook

    # register hooks on top-level layer modules (if present)
    for nm, m in model.named_modules():
        if nm in ("layer1", "layer2", "layer3", "layer4"):
            hooks.append(m.register_forward_hook(make_hook(nm)))

    with torch.no_grad():
        # ensure model and input are on same device
        device_of_model = (
            next(model.parameters()).device
            if any(p.requires_grad for p in model.parameters())
            else x.device
        )
        _ = model(x.to(device_of_model))

    for h in hooks:
        h.remove()

    for k, v in shapes.items():
        print(f"{k} output shape: {v}")


# MLP
class MLP(nn.Module):
    def __init__(self, run):
        super().__init__()
        dict = OrderedDict()
        dict.update({"flatten": nn.Flatten()})
        if run.model.num_layers == 1:
            dict.update(
                {
                    "layer1": nn.Linear(
                        run.dataset.input_dim,
                        run.dataset.output_dim,
                        bias=run.model.use_bias,
                    )
                }
            )
        else:
            for i in range(run.model.num_layers):
                if i == 0:
                    dict.update(
                        {
                            "layer"
                            + str(i + 1): nn.Linear(
                                run.dataset.input_dim,
                                run.model.width,
                                bias=run.use_bias,
                            )
                        }
                    )
                    # if not experiment_parameters["linear"]:
                    dict.update({"activation" + str(i + 1): run.model.activation_fn()})
                elif i == run.model.num_layers - 1:
                    dict.update(
                        {
                            "layer"
                            + str(i + 1): nn.Linear(
                                run.model.width,
                                run.dataset.output_dim,
                                bias=run.use_bias,
                            )
                        }
                    )
                else:
                    dict.update(
                        {
                            "layer"
                            + str(i + 1): nn.Linear(
                                run.model.width,
                                run.model.width,
                                bias=run.use_bias,
                            )
                        }
                    )
                    # if not experiment_parameters["linear"]:
                    dict.update({"activation" + str(i + 1): run.model.activation_fn()})
        self.model = nn.Sequential(dict)

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def initialize_weights(self, initialization="default"):
        for m in self.modules():
            if isinstance(m, nn.Linear) and initialization == "xavier_normal":
                nn.init.xavier_normal_(m.weight, 1)

            if isinstance(m, nn.Linear) and initialization == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, 1)

            if isinstance(m, nn.Linear) and initialization == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_in")

            if isinstance(m, nn.Linear) and initialization == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode="fan_in")


# CNN
class CNN(nn.Module):
    def __init__(self, run):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                run.dataset.image_colour_channels,
                32,
                bias=run.use_bias,
                kernel_size=3,
                padding=1,
            ),
            run.model.activation_fn(),
            run.model.pooling(run.model.window_size),
            nn.Conv2d(
                32,
                32,
                bias=run.use_bias,
                kernel_size=3,
                padding=1,
            ),
            run.model.activation_fn(),
            run.model.pooling(run.model.window_size),
            nn.Flatten(),
            nn.Linear(
                32
                * int(run.dataset.image_height / 4)
                * int(run.dataset.image_width / 4),
                run.dataset.output_dim,
                bias=run.use_bias,
            ),
        )

    def forward(self, x):
        predictions = self.model(x)
        return predictions


def _get_fanin_fanout(module):
    """Return (fanin, fanout) for nn.Linear or nn.Conv2d, else (None, None)."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if isinstance(module, nn.Conv2d):
        kh, kw = (
            module.kernel_size
            if isinstance(module.kernel_size, tuple)
            else (module.kernel_size, module.kernel_size)
        )
        return module.in_channels * kh * kw, module.out_channels * kh * kw
    return None, None


def select_model(run, device):
    model_type = run.model.model_type
    if model_type == "MLP":
        model = MLP(run)
    elif model_type == "CNN":
        model = CNN(run)
    elif model_type == "vgg11":
        model = torchvision.models.vgg11(
            num_classes=run.dataset.output_dim,
            dropout=0.0,
        )
        if run.dataset.image_colour_channels == 1:
            model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
            model.features[2] = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=1)

        if run.use_bias == False:
            print("Removing biases from VGG11")
            in_feat = model.classifier[6].in_features
            out_feat = model.classifier[6].out_features

            # Replace with a new Linear layer where bias=False
            model.classifier[6] = nn.Linear(in_feat, out_feat, bias=False)

    elif model_type == "resnet34":
        model = torchvision.models.resnet34(
            num_classes=run.dataset.output_dim, norm_layer=nn.Identity
        )
        if run.dataset.image_colour_channels == 1:
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        if run.use_bias == False:
            # Remove bias from final fc layer
            if hasattr(model, "fc"):
                in_features = model.fc.in_features
                out_features = model.fc.out_features
                model.fc = torch.nn.Linear(in_features, out_features, bias=False)

    elif model_type == "densenet121":
        model = torchvision.models.densenet121(
            num_classes=run.dataset.output_dim, drop_rate=0.0
        )
        if run.dataset.image_colour_channels == 1:
            model.features.conv0 = torch.nn.Conv2d(
                1, 64, kernel_size=3, stride=2, padding=1, bias=False
            )
            model.features.pool0 = torch.nn.MaxPool2d(
                kernel_size=5, stride=1, padding=0
            )
    elif model_type == "wide_resnet50_2":
        model = torchvision.models.resnet.wide_resnet50_2(
            num_classes=run.dataset.output_dim, norm_layer=nn.Identity
        )
        if run.dataset.image_colour_channels == 1:
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
    elif model_type == "tinyVIT":
        if (
            run.dataset.name == "CIFAR10"
            or run.dataset.name == "CIFAR100"
            or run.dataset.name == "SVHN"
        ):
            model = vit.SimpleViT(
                image_size=run.dataset.image_height,
                patch_size=8,
                num_classes=run.dataset.output_dim,
                dim=256,
                depth=4,
                heads=8,
                mlp_dim=512,
            )
        elif run.dataset.name == "EMNIST":
            model = vit.SimpleViT(
                image_size=run.dataset.image_height,
                patch_size=7,
                num_classes=run.dataset.output_dim,
                dim=256,
                depth=4,
                heads=8,
                mlp_dim=512,
            )
    else:
        raise ValueError("Not a valid model")

    summary(model)
    print("How many GPU's? ", torch.cuda.device_count(), flush=True)
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!", flush=True)
        model = nn.DataParallel(model)
    model.to(device)
    return model
