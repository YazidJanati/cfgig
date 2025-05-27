# copy/paste of selected parts from edm2 code base

import PIL
import pickle

import torch
import numpy as np

from edm2 import dnnlib
from edm2.torch_utils import misc

# ----------------------------------------------------------------------------
# Abstract base class for feature detectors.


class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x):  # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError  # to be overridden by subclass


# ----------------------------------------------------------------------------
# InceptionV3 feature detector.
# This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz


class InceptionV3Detector(Detector):
    def __init__(self):
        super().__init__(feature_dim=2048)
        url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

    def __call__(self, x):
        return self.model.to(x.device)(x, return_features=True)


# ----------------------------------------------------------------------------
# DINOv2 feature detector.
# Modeled after https://github.com/layer6ai-labs/dgm-eval


class DINOv2Detector(Detector):
    def __init__(self, resize_mode="torch"):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings

        warnings.filterwarnings("ignore", "xFormers is not available")
        torch.hub.set_dir(dnnlib.make_cache_dir_path("torch_hub"))
        self.model = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitl14",
            trust_repo=True,
            verbose=False,
            skip_validation=True,
        )
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if (
            self.resize_mode == "pil"
        ):  # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack(
                [
                    np.uint8(
                        PIL.Image.fromarray(xx, "RGB").resize(
                            (224, 224), PIL.Image.Resampling.BICUBIC
                        )
                    )
                    for xx in x
                ]
            )
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif (
            self.resize_mode == "torch"
        ):  # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(
                x.to(torch.float32), size=(224, 224), mode="bicubic", antialias=True
            )
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)


def load_stats(path, verbose=True):
    if verbose:
        print(f"Loading feature statistics from {path} ...")
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith(
            ".npz"
        ):  # backwards compatibility with https://github.com/NVlabs/edm
            return {"fid": dict(np.load(f))}
        return pickle.load(f)
