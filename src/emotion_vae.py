import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(
        self,
        z_dim=256,
        cond_dim=2,
        img_channels=3,
        img_size=128,
        latent_dim=128,
        generator_only=False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.generator_only = generator_only
        self.latent_fc = nn.Linear(self.latent_dim, self.latent_dim)

        # encoder is optional (for generator-only we skip it)
        if not generator_only:
            self.encoder = nn.Sequential(
                nn.Conv2d(img_channels, 32, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * (img_size // 8) ** 2, 512),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(512, latent_dim)
            self.fc_logvar = nn.Linear(512, latent_dim)

        self.emotion_fc = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Tanh()
        )

        self.mix_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2),
            nn.ReLU()
        )

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 128 * 8 * 8),
            nn.ReLU(inplace=True)
        )


        # ---------------------------------------------
        # Decoder architecture
        # ---------------------------------------------

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),

            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(16, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),  # final activation → outputs [-1, 1]
        )

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)
        
        self.output_gain = nn.Parameter(torch.ones(1) * 1.5)

    # Encoder
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decoder
    def decode(self, z, cond):
        """
        Decode latent z and condition vector cond into an image.
        Includes mild normalization and activation stabilization.
        """
        z_audio = cond[:, :-self.cond_dim]      # 256-dim audio embedding
        va = cond[:, -self.cond_dim:]           # 2-dim valence/arousal

        # Embed condition (valence/arousal)
        cond_embed = self.emotion_fc(va)        # → (batch, cond_dim*16)

        # Map latent and modulate by condition
        z_embed = self.latent_fc(z)
        # Modulate latent by first latent_dim portion of cond embedding (broadcast safe)
        cond_slice = cond_embed[:, : self.latent_dim]
        z_mod = z_embed * (1 + 0.25 * cond_slice)

        # Fuse latent + condition and decode
        h = torch.cat([z_mod, cond_embed], dim=1)
        h = self.mix_fc(h)
        h = F.dropout(h, p=0.1, training=self.training)

        # Stabilize hidden magnitudes before decoding
        h = torch.tanh(h / (h.abs().mean(dim=1, keepdim=True) + 1e-6))

        # Feed to fully-connected decoder head
        h = self.fc_decode(h)

        # Pass through convolutional decoder
        x = self.decoder(h)
        if not hasattr(self, "output_gain"):
            self.output_gain = nn.Parameter(torch.ones(1) * 1.5)
        # Final output stabilization — keep within [-1, 1]
        x = torch.tanh(x * self.output_gain)
        x = x.clamp(-1.0, 1.0)

        return x

    # Forward pass
    def forward(self, x=None, cond=None):
        if self.generator_only:
            z = torch.randn((cond.size(0), self.latent_dim), device=cond.device)
            return self.decode(z, cond), None, None

        mu, logvar = self.encode(x)
        logvar = logvar.clamp(-5.0, 5.0)
        z = self.reparameterize(mu, logvar)

        # Slight latent noise jitter to encourage diversity
        if self.training:
            z = z + 0.05 * torch.randn_like(z)

        recon = self.decode(z, cond)
        return recon, mu, logvar