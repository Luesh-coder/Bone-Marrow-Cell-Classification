"""
Models (proposal slide 8).

    baseline         small CNN trained from scratch: 5 x [conv3x3 -> BN -> ReLU ->
                     max-pool], global average pool, small dropout head (~1M params).
                     Shows what the data alone can teach.
    resnet50         ImageNet-pretrained ResNet-50 (torchvision IMAGENET1K_V2), final
                     layer replaced with a dropout + 21-class head.
    efficientnet_b0  lighter pretrained fallback (slide 13), same head treatment.

Fine-tuning helpers:
    set_trainable(model, arch, unfreeze)   "head" | "last_block" | "all"
    freeze_bn_stats(model)                 call after model.train(): keeps BatchNorm
                                           running stats of frozen layers fixed
    param_groups(...)                      head at lr, pretrained layers at
                                           lr * backbone_lr_mult; no weight decay on
                                           biases and BatchNorm
"""

import torch
from torch import nn
from torchvision import models as tvm


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=21, widths=(32, 64, 128, 256, 256), dropout=0.3, hidden=128):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_resnet50(num_classes=21, pretrained=True, dropout=0.3):
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = tvm.resnet50(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m


def build_efficientnet_b0(num_classes=21, pretrained=True, dropout=0.3):
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.efficientnet_b0(weights=weights)
    in_features = m.classifier[-1].in_features
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return m


def head_and_last_block(model, arch):
    """(classifier head, last feature block) of each architecture."""
    if arch == "resnet50":
        return model.fc, model.layer4
    if arch == "efficientnet_b0":
        return model.classifier, model.features[7:]   # last MBConv stage + final 1x1 conv
    if arch == "baseline":
        return model.head, model.features[-4:]
    raise ValueError(f"unknown architecture {arch!r}")


def set_trainable(model, arch, unfreeze="all"):
    """Freeze everything except the head ("head"), head + last block ("last_block"),
    or nothing ("all"). The from-scratch baseline is always fully trainable."""
    if unfreeze == "all" or arch == "baseline":
        for p in model.parameters():
            p.requires_grad = True
        return model
    head, last = head_and_last_block(model, arch)
    for p in model.parameters():
        p.requires_grad = False
    for module in (head, last) if unfreeze == "last_block" else (head,):
        for p in module.parameters():
            p.requires_grad = True
    return model


def freeze_bn_stats(model):
    """Put BatchNorm layers whose parameters are frozen into eval mode, so their
    ImageNet running statistics don't drift. Call after every model.train()."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()
    return model


def param_groups(model, arch, lr, weight_decay, backbone_lr_mult=0.1, pretrained=True):
    """Optimizer parameter groups: head vs backbone (lower LR when pretrained),
    each split into decay / no-decay (biases and normalization weights)."""
    head, _ = head_and_last_block(model, arch)
    head_ids = {id(p) for p in head.parameters()}
    backbone_lr = lr * backbone_lr_mult if (pretrained and arch != "baseline") else lr
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "backbone"
        decay = p.ndim > 1
        key = (part, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr if part == "head" else backbone_lr,
                           "weight_decay": weight_decay if decay else 0.0,
                           "name": f"{part}{'' if decay else '_no_decay'}"}
        groups[key]["params"].append(p)
    return list(groups.values())


def build_model(cfg, num_classes):
    """Model for a TrainConfig, with the requested layers unfrozen."""
    if cfg.model == "baseline":
        m = BaselineCNN(num_classes, dropout=cfg.dropout)
    elif cfg.model == "resnet50":
        m = build_resnet50(num_classes, cfg.pretrained, cfg.dropout)
    elif cfg.model == "efficientnet_b0":
        m = build_efficientnet_b0(num_classes, cfg.pretrained, cfg.dropout)
    else:
        raise ValueError(f"unknown model {cfg.model!r}")
    return set_trainable(m, cfg.model, cfg.unfreeze)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    from config import TrainConfig

    for arch in ("baseline", "resnet50", "efficientnet_b0"):
        m = build_model(TrainConfig(model=arch, pretrained=False), 21)
        total, _ = count_params(m)
        out = m(torch.zeros(2, 3, 224, 224))
        print(f"{arch:<16} {total / 1e6:6.2f}M params, output {tuple(out.shape)}")
