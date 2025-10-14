import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc_mu = nn.Linear(512, latent_dim)

        # More powerful decoder with adaptive upsampling
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 4, 4)),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(128, 645), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


        # final adaptive resize to match Mel shape
        self.final_resize = nn.Upsample(size=(128, 645), mode="bilinear", align_corners=False)

    def encode(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        return self.fc_mu(z)

    def decode(self, z):
        recon = self.decoder(z)
        recon = self.final_resize(recon)
        return recon

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.fc_mu(z)
        z_min, _ = z.min(dim=1, keepdim=True)
        z_max, _ = z.max(dim=1, keepdim=True)
        z = (z - z_min) / (z_max - z_min + 1e-8)

        recon = self.decoder(z)

        recon = torch.clamp(recon, 0, 1)

        return recon, z
