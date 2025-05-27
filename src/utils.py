import yaml
import PIL.Image
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
import torch
import torch.distributions as dist
import random


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_pil_image(x: torch.Tensor) -> List[PIL.Image.Image]:
    # NOTE: this steps is irrelevant for EDM model
    # since the out of the decoder is in [0, 255]
    if x.dtype != torch.uint8:
        x = x.clamp(-1.0, 1.0)
        x = (x + 1.0) * 127.5

    sample = x.cpu().permute(0, 2, 3, 1)

    sample = sample.numpy().astype(np.uint8)
    if sample.shape[0] == 1:
        img_pil = PIL.Image.fromarray(sample[0])
        return img_pil
    else:
        return [PIL.Image.fromarray(s) for s in sample]


def display(x, save_path=None, title=None):
    img_pil = get_pil_image(x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path + ".png")


def display_dic(dic, title, nimages_line=3):
    n_images = len(dic)
    nimages_col = np.minimum(1, n_images % nimages_line) + n_images // nimages_line
    fig, ax = plt.subplots(
        nrows=nimages_col, ncols=nimages_line, figsize=(30, nimages_col * 10)
    )
    if ax.ndim == 1:
        ax = [ax]
    if type(title) is str:
        fig.suptitle(title + "\n", fontsize=40, y=1.2)

    plt.subplots_adjust(wspace=0.001)

    for axis in fig.get_axes():
        axis.axis("off")

    for idx, im in enumerate(dic):
        img_pil = get_pil_image(dic[im])
        ax[idx // nimages_line][idx % nimages_line].imshow(img_pil)
        ax[idx // nimages_line][idx % nimages_line].set_title(im, fontsize=35)

    plt.show()
    plt.close()


def display_grid(dic, save_path=None, title=None):
    fig, ax = plt.subplots(1, len(dic), figsize=(20, 10))
    for idx, im in enumerate(dic):
        img_pil = get_pil_image(dic[im])
        ax[idx].imshow(img_pil)
        ax[idx].set_title(im)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    plt.close()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def cosine_beta_scheduler(timesteps, s=0.008):
    """
    Create a cosine schedule for the noise variance (beta) as proposed in the
    "Improved Denoising Diffusion Probabilistic Models" paper.

    Args:
        timesteps (int): Number of timesteps in the diffusion process
        s (float): Small offset to prevent beta from being too small near t=0

    Returns:
        torch.Tensor: Beta schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clipping to ensure numerical stability
    return torch.clamp(betas, 0.0001, 0.9999)


def display_audio(track, sample_rate=16_000, normalize=False):
    """Display audio in interactive mode."""
    from IPython.display import Audio, display as ipy_display

    ipy_display(Audio(track, rate=sample_rate, normalize=normalize))
