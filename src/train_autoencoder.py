"""
train_autoencoder.py
--------------------
Train the ResNetAutoencoder on Mel spectrogram .npy files.

You can:
- Run from CLI:  python src/train_autoencoder.py
- Or call from notebook: from src.train_autoencoder import train_autoencoder
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import MelSpectrogramDataset, pad_or_crop_batch
from mel_autoencoder import ResNetAutoencoder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from config import (
    PROCESSED_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_DIR,
)
torch.set_float32_matmul_precision("high")


class AutoencoderTrainer(pl.LightningModule):
    def __init__(self, lr=LEARNING_RATE, latent_dim=256):
        super().__init__()
        self.model = ResNetAutoencoder(latent_dim)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, _ = self.model(x)
        loss = self.loss_fn(recon, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, _ = self.model(x)
        loss = self.loss_fn(recon, x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_autoencoder(latent_dim=256, max_epochs=EPOCHS):
    """
    Trains the ResNetAutoencoder directly from a notebook cell.

    Returns:
        model (AutoencoderTrainer): trained LightningModule
        trainer (pl.Trainer): trainer instance with logs
    """
    pl.seed_everything(42)

    dataset = MelSpectrogramDataset(PROCESSED_DIR)
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=pad_or_crop_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=pad_or_crop_batch
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = MODEL_DIR / "autoencoder_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="resnet_autoencoder-{epoch:02d}-{val_loss:.4f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )

    early_stop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    logger = CSVLogger(save_dir=str(MODEL_DIR / "logs"), name="autoencoder")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
    )

    model = AutoencoderTrainer(latent_dim=latent_dim)
    trainer.fit(model, train_loader, val_loader)

    final_model_path = MODEL_DIR / "autoencoder" / "resnet_autoencoder_final.pt"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), final_model_path)
    print(f"[INFO] Model saved to {final_model_path}")

    return model, trainer


if __name__ == "__main__":
    train_autoencoder()
