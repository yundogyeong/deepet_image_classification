import torch
import torch.nn as nn
from functools import partial

class ConvBnAct(nn.Module):
    def __init__(
        self, in_feat:int, out_feat:int, kernel_size:int, stride:int, padding:int,
        bn:bool=True,
        act:bool=True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, bias=not bn
        )
        if bn:
            self.bn = nn.BatchNorm2d(out_feat)
        else:
            self.bn = None
        if act:
            self.act = nn.ReLU()
        else:
            self.act = None
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class downsample(nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.LazyConv2d(feat, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(feat)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.body(x)

class basicResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, half:bool):
        super().__init__()
        self.conv_1 = ConvBnAct(in_feat, out_feat, 3, 1, 1)
        if not half:
            self.conv_2 = ConvBnAct(out_feat, out_feat, 3, 1, 1, bn=True, act=False)
            self.downsample = None
        else:
            # if half
            self.conv_2 = ConvBnAct(out_feat, out_feat, 3, 2, 1,  bn=True, act=False)
            self.downsample = downsample(out_feat)

        self.final_act = nn.ReLU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        return self.final_act(x)
 
# if resnet depth >= 50
class bottleneck(nn.Module):
    pass


class RESNET18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBnAct(3, 64, 7, 2, 3)
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            basicResBlock(64, 64, False),
            basicResBlock(64, 64, False)
        )
        self.stage2 = nn.Sequential(
            basicResBlock(64, 128, True),
            basicResBlock(128, 128, False)
        )
        self.stage3 = nn.Sequential(
            basicResBlock(128, 256, True),
            basicResBlock(256, 256, False),
        )
        self.stage4 = nn.Sequential(
            basicResBlock(256, 512, True),
            basicResBlock(512, 512, False),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # random_input = torch.randn(1, 3, 224, 224)
    # vanilla_conv = ConvBnAct(3, 6, 3, 1, 1, False, True)
    # random_output = vanilla_conv(random_input)
    # print(random_output.shape)
    
    # random_input = torch.randn(1, 64, 112, 112)
    # first_resblock = basicResBlock(64, False)
    # random_output = first_resblock(random_input)
    # print(random_output.shape)
    
    # second_resblock = basicResBlock(64, True)
    # random_output = second_resblock(random_output)
    # print(random_output.shape)
    random_input = torch.randn(1, 3, 224, 224)
    
    # test resnet18
    resnet_18 = RESNET18()
    resnet_18_output = resnet_18(random_input)