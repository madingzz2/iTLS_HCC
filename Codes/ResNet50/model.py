import torch.nn as nn
from torchvision.models import resnet50

class ResNet50_3slice(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained)
        # 首层改 3 通道，最后改分类头
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def forward_features(self, x):
        """返回 avgpool 2048-d"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return x.view(x.size(0), -1)