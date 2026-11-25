"""CIFAR-stem ResNet with the approved depth/width choices."""
import torch
from torch import nn
from .config import Candidate


class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.shortcut = nn.Identity() if stride == 1 and incoming == outgoing else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm2d(outgoing))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, candidate: Candidate, classes=10):
        super().__init__()
        if classes < 2:
            raise ValueError("Classification requires at least two classes")
        width = candidate.width
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        stages = []
        incoming = width
        for stage, count in enumerate(candidate.blocks):
            outgoing = width * 2 ** stage
            blocks = []
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(BasicBlock(incoming, outgoing, stride))
                incoming = outgoing
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(candidate.dropout)
        self.classifier = nn.Linear(incoming, classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.dropout(self.pool(self.stages(self.stem(x))).flatten(1)))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def make_optimizer(model, candidate):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": candidate.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if candidate.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=candidate.learning_rate, momentum=0.9, nesterov=False)
    return torch.optim.AdamW(groups, lr=candidate.learning_rate, betas=(0.9, 0.999), eps=1e-8)

