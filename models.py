import torch
import torch.nn as nn


class ASoftmax(nn.Module):
    def __init__(self, m=4):
        super().__init__()
    
    def forward(self, x, w):
        pass


class ResidualModule(nn.Module):
    def __init__(self, in_ch, bottleneck):
        super().__init__()
        out_ch = in_ch
        convs = []
        convs += [nn.Sequential(
            nn.Conv2d(in_ch, bottleneck, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(bottleneck), 
            nn.PReLU(bottleneck)
        )]
        convs += [nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(bottleneck), 
            nn.PReLU(bottleneck)
        )]
        convs += [nn.Sequential(
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(out_ch) 
        )]
        self.model = nn.Sequential(*convs)
    
    def forward(self, x):
        z = self.model(x)
        return x+z


class ResNet50(nn.Module):
    """
        Input: (NxCxHxW)
        Output: (Nx128)
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 10000
        convs = []
        convs += [nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        )]
        convs += [nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.PReLU(128)
        )]
        for _ in range(3):
            convs += [ResidualModule(128, 64)]
        convs += [nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.PReLU(256)
        )]
        for _ in range(4):
            convs += [ResidualModule(256, 128)]
        convs += [nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.PReLU(512)
        )]
        for _ in range(6):
            convs += [ResidualModule(512, 256)]
        convs += [nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(1024), 
            nn.PReLU(1024)
        )]
        convs += [nn.AdaptiveAvgPool2d(1)]
        for _ in range(3):
            convs += [ResidualModule(1024, 512)]
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.PReLU(1)
        )
        self.out = nn.Linear(512, self.num_classes, bias=False)
        # self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), 1024)
        x = self.fc(x)
        if self.training:
            with torch.no_grad():
                self.out.weight.div_(self.out.weight.norm(p=2, dim=1, keepdim=True))
        return x