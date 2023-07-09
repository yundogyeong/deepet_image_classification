import torch
import torch.nn as nn

# VGG 블록 정의
class VGGBLock(nn.Module):
    def __init__(self, num_convs, out_channels):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.LazyConv2d(out_channels, kernel_size=3, stride=1, padding=1)
            )
            layers.append(
                nn.LazyBatchNorm2d()
            )
            layers.append(
                nn.ReLU()
            )
        layers.append(
            # kernel=2, stride=2 이므로 H, W 값이 각각 0.5*H, 0.5*W로 축소
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
# VGG 네트워크 정의
class VGG(nn.Module):
    # 논문에 제시된 기본 설계
    default_config = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    
    def __init__(self, cfg=None, num_classes=1000):
        super().__init__()
        # config파일이 따로 주어지지 않는다면 기본값을 사용
        cfg = self.default_config if cfg is None else cfg
        conv_blks = []
        
        # config의 내용에 따라 네트워크를 구성
        for (num_convs, out_channels) in cfg:
            conv_blks.append(VGGBLock(num_convs, out_channels))
        
        # Iterable unpacking
        self.backbone = nn.Sequential(
            *conv_blks
        )
        
        # 분류를 위한 헤드 부분
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.1),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.1),
            nn.LazyLinear(num_classes)
        )
    
    def forward(self, x):
        feature = self.backbone(x)
        preds = self.head(feature)
        return preds
    
if __name__ == "__main__":
    net = VGG()
    random_input = torch.randn(1, 3, 224, 224)
    b = net(random_input)
    print(b.shape)