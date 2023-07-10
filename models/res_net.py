import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class ResidualBlock18(nn.Module):
    def __init__(self, in_channels, out_channels, strides):
        super(ResidualBlock18, self).__init__()

        self.conv_block = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        if x.shape != residual.shape: #shape가 다를 때 skip connection을 위한 down sampling
            x += self.downsample(residual)
        else:
            x += residual
        x = self.act(x)
        return x


class ResidualBlock50(nn.Module): #bottle neck
    def __init__(self, in_channels, out_channels, strides):
        super(ResidualBlock50, self).__init__()

        self.conv_block = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1),
            nn.BatchNorm2d(out_channels * 4)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=strides),
            nn.BatchNorm2d(out_channels * 4)
        )

        self.final_act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        if x.shape != residual.shape: #shape가 다를 때 skip connection을 위한 down sampling
            x += self.downsample(residual)
        else:
            x += residual
        x = self.final_act(x)
        return x


class ResidualNetwork(nn.Module):
    def __init__(self, num_class, num_layers):
        super(ResidualNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch = nn.BatchNorm2d(64)
        self.re = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if num_layers == 18:  # ResNet 18
            block_name = ResidualBlock18
            self.res_block1 = self.iter_layer(block_name,64, 64, 1, 2)
            self.res_block2 = self.iter_layer(block_name,64, 128, 2, 2)
            self.res_block3 = self.iter_layer(block_name,128, 256, 2, 2)
            self.res_block4 = self.iter_layer(block_name,256, 512, 2, 2)
            self.fc = nn.Linear(512, num_class)

        if num_layers == 50:  # ResNet 50
            block_name = ResidualBlock50
            self.res_block1 = self.iter_layer(block_name,64, 64, 1, 3)
            self.res_block2 = self.iter_layer(block_name,256, 128, 2, 4)
            self.res_block3 = self.iter_layer(block_name,512, 256, 2, 6)
            self.res_block4 = self.iter_layer(block_name,1024, 512, 2, 3)
            self.fc = nn.Linear(2048, num_class)

    def iter_layer(self, block_name ,in_channels, out_channels, strides, num):
        blk = []
        blk.append(block_name(in_channels, out_channels, strides))
        for _ in range(1, num):
            blk.append(block_name(out_channels, out_channels, strides=1))
        return nn.Sequential(*blk)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch(x)
        x = self.re(x)
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    random_input = torch.randn(1, 3, 224, 224)
    first_residual_block = ResidualNetwork(1000, 18)
    random_output = first_residual_block(random_input)
    print(random_output.shape)
    model = ResidualNetwork(1000, 18)
    #print(model)
