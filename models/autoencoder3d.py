import torch
import torch.nn as nn
import torch.nn.functional as F

class AE3D(nn.Module):
    """Tiny 3D-Conv autoencoder for (B,1,16,112,112)."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),  # 16x8x56x56
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), # 32x4x28x28
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), # 64x2x14x14
            nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), # 32x4x28x28
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1), # 16x8x56x56
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, 4, stride=2, padding=1),  # 1x16x112x112
            nn.Sigmoid()
        )

    def forward(self, x): return self.dec(self.enc(x))

def reconstruction_error(x, xhat):
    return F.mse_loss(xhat, x, reduction='none').mean(dim=[1,2,3,4])
