"""
visualize_cvae.py
-----------------
Utility to visualize the Conditional VAE's generated abstract art
across a grid of valence and arousal values.

Usage:
    from src.visualize_cvae import show_emotion_grid
    show_emotion_grid("models/vae/conditional_vae.pt")
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from emotion_vae import ConditionalVAE
from PIL import Image, ImageEnhance


def show_emotion_grid(
    model_path: str = "../models/vae/conditional_vae.pt",
    z_dim: int = 128,
    cond_dim: int = 2,
    latent_dim: int = 128,
    img_size: int = 128,
    n_steps: int = 5,
    device: str = None,
    contrast_gain: float = 1.25,   # overall output contrast
    saturation_gain: float = 1.6,  # color vibrancy
):
    """
    Displays a grid of generated images from the Conditional VAE
    with valence (x-axis) and arousal (y-axis) variation.

    Args:
        model_path: path to trained CVAE weights
        z_dim: audio embedding dimension
        cond_dim: conditioning dimension (valence + arousal)
        latent_dim: latent dimension of the VAE
        img_size: output image size
        n_steps: number of values for valence/arousal grid
        device: 'cuda' or 'cpu'
        contrast_gain: multiplier for decoder output contrast
        saturation_gain: post-color enhancement multiplier
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Load model
    # ------------------------------
    print(f"[INFO] Loading ConditionalVAE from: {model_path}")
    gen = ConditionalVAE(
        z_dim=z_dim, cond_dim=cond_dim,
        img_channels=3, img_size=img_size,
        latent_dim=latent_dim
    )
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.to(device).eval()

    plot_vals = np.linspace(2, 9, n_steps)   # for axis labels (2–9)
    model_vals = plot_vals / 10.0             # model expects [0–1]

    # Create figure
    rows, cols = len(model_vals), len(model_vals)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    with torch.no_grad():
        for i, ar in enumerate(reversed(model_vals)):  # Arousal (y-axis)
            for j, va in enumerate(model_vals):        # Valence (x-axis)
                cond_vec = np.concatenate([np.zeros(z_dim, dtype=np.float32), [va, ar]], axis=0)
                cond = torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0)
                z = torch.zeros((1, latent_dim), device=device)

                # Decode image safely
                img = gen.decode(z, cond).detach().cpu()
                img = torch.tanh(img * contrast_gain)      # soft contrast expansion
                img = ((img + 1) / 2).clamp(0, 1)          # normalize to [0,1]
                img_np = img[0].permute(1, 2, 0).numpy()

                # Convert to PIL and apply visual enhancements
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                img_pil = ImageEnhance.Color(img_pil).enhance(saturation_gain)
                img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast_gain)

                # Convert back to numpy for plotting
                img_np = np.asarray(img_pil) / 255.0

                # Display
                axes[i, j].imshow(img_np)
                axes[i, j].axis("off")

                if i == rows - 1:
                    axes[i, j].set_title(f"V={plot_vals[j]:.1f}", fontsize=8)
            axes[i, 0].set_ylabel(f"A={plot_vals[::-1][i]:.1f}", fontsize=8)

    plt.suptitle("Conditional VAE: Valence (x) × Arousal (y)", fontsize=14)
    plt.show()
