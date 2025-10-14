import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from pathlib import Path

from emotion_vae import ConditionalVAE
from vae_dataset import EmbeddingsToArtDataset
from config import MODEL_DIR, BATCH_SIZE, EPOCHS

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import pandas as pd
import numpy as np
from pytorch_msssim import ssim  

# Enhanced CVAE Loss Function
def cvae_loss_function(
    recon_x, x, mu, logvar, cond, step, max_steps,
    beta=0.0015, base_color_weight=0.25, base_ssim_weight=0.05, base_cond_weight=0.15, output_gain=None
):
    """
    Stable, differentiable CVAE loss with KL warm-up, tanh decoder, and gradient-safe guards.
    """
    max_beta = 1e-4
    warmup_fraction = 0.5
    warmup_steps = max_steps * warmup_fraction
    beta = max_beta * min(float(step) / (warmup_steps + 1e-8), 1.0)

        # Clamp recon range for stability
    recon_x = recon_x.tanh().clamp(-1.0, 1.0)
    x = x.clamp(-1.0, 1.0)

    # Reconstruction
    recon_loss = (0.8 * F.l1_loss(recon_x, x) + 0.2 * F.mse_loss(recon_x, x)) * 5.0

    # Color and SSIM
    mean_true = x.mean((2, 3))
    mean_recon = recon_x.mean((2, 3))
    color_loss = F.mse_loss(mean_recon, mean_true)

    try:
        ssim_val = ssim(recon_x, x, data_range=2.0, size_average=True)
    except Exception:
        ssim_val = torch.tensor(0.0, device=x.device)
    perceptual_loss = (1 - ssim_val).clamp(0, 1)

    # KL (with clipping)
    logvar = logvar.clamp(-5, 5)
    kl_loss_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss_raw = kl_loss_raw / x.size(0)
    progress = min(float(step) / (float(max_steps) + 1e-8), 1.0)
    kl_weight = beta * (0.5 + 0.5 * progress)
    kl_loss = kl_weight * kl_loss_raw

    # Conditional brightness correlation
    va = cond[:, -2:]
    bright = recon_x.mean((1, 2, 3)).unsqueeze(1)
    va_norm = (va - va.mean(0, keepdim=True)) / (va.std(0, keepdim=True) + 1e-8)
    bright_norm = (bright - bright.mean()) / (bright.std() + 1e-8)
    corr = torch.mean(va_norm * bright_norm).clamp(-1, 1)
    cond_loss = ((1.0 - corr) ** 2).clamp(0, 4)

    # Variance match
    var_loss = F.mse_loss(recon_x.var((2,3)), x.var((2,3)))

    total_loss = (
        recon_loss
        + base_color_weight * color_loss
        + base_ssim_weight * perceptual_loss
        + 0.05 * var_loss
        + base_cond_weight * cond_loss
        + kl_loss
    )
    
    if output_gain is not None:
        # encourage it to stay near ~2.0, but allow learning
        target_gain = 2.0
        gain_reg = 0.001 * (output_gain.mean() - target_gain).pow(2)
        total_loss += gain_reg

    if not torch.isfinite(total_loss):
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=10.0, neginf=10.0)

    return total_loss, {
        "recon": float(recon_loss.detach().cpu()),
        "color": float(color_loss.detach().cpu()),
        "ssim": float(perceptual_loss.detach().cpu()),
        "cond": float(cond_loss.detach().cpu()),
        "kl": float(kl_loss_raw.detach().cpu()),
        "progress": progress,
    }


# Lightning Module
class CVAETrainer(pl.LightningModule):
    def __init__(self, z_dim=256, cond_dim=2, img_size=128, latent_dim=128, lr=3e-4, beta=0.0005):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConditionalVAE(
            z_dim=z_dim,
            cond_dim=cond_dim,
            img_channels=3,
            img_size=img_size,
            latent_dim=latent_dim,
        )
        self.lr = lr
        self.beta = beta

    def training_step(self, batch, batch_idx):
        x, cond, _ = batch
        x_recon, mu, logvar = self.model(x, cond)

        loss, loss_dict = cvae_loss_function(
            x_recon, x, mu, logvar, cond,
            step=self.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            beta=self.beta,
            output_gain=self.model.output_gain
        )

        # Log loss components
        self.log_dict({
            "train_loss": loss,
            "train_recon": loss_dict["recon"],
            "train_color": loss_dict["color"],
            "train_ssim": loss_dict["ssim"],
            "train_cond": loss_dict["cond"],
            "train_kl": loss_dict["kl"],
            "output_gain": float(self.model.output_gain.detach().cpu().mean())
        }, prog_bar=True, on_epoch=True)
        
        if not torch.isfinite(loss):
            self.log("train_loss_nan", 1)
            loss = torch.zeros_like(loss)
        return loss

    def validation_step(self, batch, _):
        x, cond, _ = batch
        x_recon, mu, logvar = self.model(x, cond)

        loss, loss_dict = cvae_loss_function(
            x_recon, x, mu, logvar, cond,
            step=self.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            beta=self.beta,
            output_gain=self.model.output_gain
        )

        # Log validation metrics
        self.log_dict({
            "val_loss": loss,
            "val_recon": loss_dict["recon"],
            "val_color": loss_dict["color"],
            "val_ssim": loss_dict["ssim"],
            "val_cond": loss_dict["cond"],
            "val_kl": loss_dict["kl"],
            "val_output_gain": float(self.model.output_gain.detach().cpu().mean())
        }, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=2e-5)
        return optimizer


# Training Function

def train_cvae(
    embeddings_csv="../data/processed/embeddings.csv",
    img_size=128, z_dim=256, cond_dim=2, latent_dim=128,
    batch_size=BATCH_SIZE, max_epochs=EPOCHS,
    lr=3e-4, beta=0.0005
):

    pl.seed_everything(42)

    print(f"[INFO] Loading embeddings from {embeddings_csv}")
    ds = EmbeddingsToArtDataset(embeddings_csv=embeddings_csv, img_size=img_size)
    n_total = len(ds)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = MODEL_DIR / "vae_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="cvae-{epoch:02d}-{val_loss:.4f}",
        save_top_k=2, monitor="val_loss", mode="min",
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=15, min_delta=1e-4 )
    logger = CSVLogger(save_dir=str(MODEL_DIR / "logs"), name="cvae")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
    )

    model = CVAETrainer(
        z_dim=z_dim, cond_dim=cond_dim, img_size=img_size,
        latent_dim=latent_dim, lr=lr, beta=beta
    )
    trainer.fit(model, train_loader, val_loader)

    out_path = MODEL_DIR / "vae" / "conditional_vae.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), out_path)
    print(f"[INFO] CVAE saved to {out_path}")

    return model, trainer
