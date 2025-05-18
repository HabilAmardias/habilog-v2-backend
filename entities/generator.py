from torch.nn import functional as F
from torch import nn

class Stem(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=9, padding=4, stride=1),
            nn.PReLU(dim)
        )
    def forward(self,x):
        return self.model(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.PReLU(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim)
        )
    def forward(self,x):
        return self.model(x) + x

class Refinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )
    def forward(self,x,memory):
        return self.model(x) + memory
    
class ImageExpansion(nn.Module):
    def __init__(self, dim, upscale_factor):
        super().__init__()
        self.expansion = nn.Sequential(
            nn.Conv2d(dim, dim*upscale_factor, 3, padding=1),
            nn.PixelShuffle(upscale_factor//2),
            nn.PReLU(dim)
        )
    def forward(self,x):
        return self.expansion(x)

class Generator(nn.Module):
    def __init__(self, dim, num_residual=8, num_expansion=2, upscale_factor=4):
        super().__init__()
        self.stem = Stem(dim)
        self.residuals = nn.ModuleList(
            [ResidualBlock(dim) for _ in range(num_residual)]
        )
        self.refinement = Refinement(dim)
        self.expansion = nn.ModuleList(
            [ImageExpansion(dim, upscale_factor) for _ in range(num_expansion)]
        )
        self.clf = nn.Conv2d(dim, 3, kernel_size=9, padding=4)
        
    def forward(self,x):
        memory = self.stem(x)
        
        h = memory
        for layer in self.residuals:
            h = layer(h)
            
        h = self.refinement(h, memory)
        
        for layer in self.expansion:
            h = layer(h)
            
        return F.tanh(self.clf(h))