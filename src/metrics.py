import tqdm
import pickle
import subprocess
from pathlib import Path

import torch
import scipy.linalg
import numpy as np

from local_paths import LARGE_FILE_DIR
from edm2.training.dataset import ImageFolderDataset
from edm2.calculate_metrics import InceptionV3Detector, DINOv2Detector


NUM_IMAGES = 50_000  # NOTE: the number of images used in EDM2 to evaluate FID and FD
NUM_IMAGES_PRDC = 50_000  # NOTE: number of images used to compute PR and DC


class Metric:
    def __init__(
        self,
        metric: str,
        batch_size: int = 50,
        num_workers: int = 1,
        device: str = "cpu",
    ) -> None:
        # precomputed stats of real dataset were taken from EDM2
        # https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/README.md#calculating-flops-and-metrics
        self.path_precomputed_stats = LARGE_FILE_DIR / "img512.pkl"

        self.metric = metric.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        # to be set depending on the metrics
        if self.metric == "fid":
            self.model = InceptionV3Detector()
            self.dims = 2048
        elif self.metric == "fd_dinov2":
            self.model = DINOv2Detector()
            self.dims = 1024

        # compute stats of real data
        # load precomputed metrics of real data
        with open(self.path_precomputed_stats, "rb") as f:
            ref_metrics = pickle.load(f)[self.metric]
        self.mean_real, self.cov_real = ref_metrics["mu"], ref_metrics["sigma"]

    def compute_FID(self, path_imgs: str | Path, seed: int = 0) -> float:
        # compute stat of data
        mean, cov = self._compute_stats(path_imgs, seed)

        return self._calculate_frechet_distance(
            self.mean_real, self.cov_real, mean, cov
        )

    def _compute_stats(self, path_imgs, seed=0):

        dims, device = self.dims, self.device
        dtype = torch.float64

        dataset_obj = ImageFolderDataset(
            path=path_imgs, max_size=NUM_IMAGES, random_seed=seed
        )
        data_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=self.batch_size,
        )

        # estimate mean and cov
        mu_cum = torch.zeros(size=(dims,), dtype=dtype, device=device)
        cov_cum = torch.zeros(size=(dims, dims), dtype=dtype, device=device)
        n_imgs = 0

        for batch in tqdm.tqdm(data_loader):
            imgs, _ = batch
            imgs = torch.as_tensor(imgs).to(device)

            features = self.model(imgs).to(dtype)

            mu_cum += features.sum(0)
            cov_cum += features.T @ features
            n_imgs += imgs.shape[0]

        mu = mu_cum / n_imgs
        # unbiased estimator of cov
        cov = cov_cum / (n_imgs - 1) - (n_imgs / (n_imgs - 1)) * mu.ger(mu)

        return mu.cpu().numpy(), cov.cpu().numpy()

    def _calculate_frechet_distance(self, mean_real, cov_real, mean, cov):

        m = np.square(mean - mean_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(cov, cov_real), disp=False)
        value = float(np.real(m + np.trace(cov + cov_real - s * 2)))

        return value


class PRDC:
    """Compute Precision/Recall and Density/Coverage as described in [1].

    ``latent_type`` defines how the latents of the datasets are computed, either
        - inceptionv3
        - dinov2
        - vgg16

    References
    ----------
    .. [1] Naeem, Muhammad Ferjad, et al.
        "Reliable fidelity and diversity metrics for generative models."
        International conference on machine learning. PMLR, 2020.
    """

    def __init__(
        self,
        latent_type: str,
        n_neighbors: int = 5,
        batch_size: int = 50,
        n_jobs: int = 1,
        device: str = "cpu",
    ) -> None:
        # XXX to change
        self.path_ref_imgs = LARGE_FILE_DIR / "imagenet512_val"

        self.latent_type = latent_type.lower()
        self.n_neighbors = n_neighbors

        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.device = device

        # to be set depending on the metrics
        if self.latent_type == "inceptionv3":
            self.model = InceptionV3Detector()
            self.dims = 2048
        elif self.latent_type == "dinov2":
            self.model = DINOv2Detector()
            self.dims = 1024
        elif self.latent_type == "vgg16":
            self.model = LatentsVGG16(device)
            self.dims = 4096
        else:
            raise ValueError("Unknow latent type.")

    def compute_prcd(self, path_imgs: str | Path, seed: int = 0):
        # compute latents
        fake_latents = self._compute_latents(path_imgs, seed)
        ref_latens = self._compute_latents(self.path_ref_imgs, seed)

        return compute_prdc(
            ref_latens, fake_latents, nearest_k=self.n_neighbors, n_jobs=self.n_jobs
        )

    def _compute_latents(self, path_imgs, seed=0) -> np.ndarray:
        dims, device = self.dims, self.device
        dtype = torch.float32

        dataset_obj = ImageFolderDataset(
            path=path_imgs, max_size=NUM_IMAGES_PRDC, random_seed=seed
        )
        data_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=self.batch_size,
        )

        # where to store latent
        n_latents = dataset_obj._raw_idx.size
        print("**********")
        print("number of images", n_latents)
        print("seed", seed)
        print("**********")
        latents = torch.zeros((n_latents, dims), device=device, dtype=dtype)

        ptr = 0
        for batch in tqdm.tqdm(data_loader):
            imgs, _ = batch
            imgs = torch.as_tensor(imgs).to(device, dtype)
            current_batch_size = len(imgs)

            features = self.model(imgs)
            latents[ptr : ptr + current_batch_size] = features

            # update pointer
            ptr += current_batch_size

        return latents.cpu().numpy()


def get_gpu_memory_consumption(device: str) -> int:
    """Get the current gpu usage.

    Code adapted from:
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Parameters
    ----------
    device : str
        name of the device, for example: 'cuda:0'

    Returns
    -------
    usage: int
        memory usage in MB.

    Notes
    -----
    - Normally this function should be called during the execution of a scripts but
      it is possible to call it at the end as GPU computation is cached.
    """
    # get device id
    try:
        device_id = int(device.replace("cuda:", ""))
    except ValueError:
        raise ValueError(f"Expected device to be of the form 'cuda:ID', got {device}")

    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    memory = gpu_memory_map.get(device_id, None)
    if memory is None:
        available_devices = [f"cuda:{i}" for i in gpu_memory_map]
        raise ValueError(
            "Unknown device name.\n"
            f"Expected device to be {available_devices}\n"
            f"got {device}"
        )

    return memory


# ---
class LatentsVGG16(torch.nn.Module):
    """Latent computed using VGG16 imagenet features maps.

    Code adapted from
    https://github.com/Mahmood-Hussain/generative-evaluation-prdc/blob/b471eb6d4ab5993fa662a307eb3a406ad80b670b/prdc/Models.py
    """

    def __init__(self, device):
        super(LatentsVGG16, self).__init__()

        # XXX defer import
        from torchvision.models import vgg16

        self.vgg16 = vgg16(pretrained=True).to(device)
        self.vgg16 = self.vgg16.eval()

        self.features = self.vgg16.features.requires_grad_(False)
        self.fc1 = self.vgg16.classifier[0].requires_grad_(False)

    def forward(self, x):
        x = LatentsVGG16.preprocess(x.to(torch.float32))

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    @staticmethod
    def preprocess(x: torch.Tensor):
        """Processing VGG16 preprocessing of the image before inference.

        The preprocessing of the image is the same as the one applied to Dinov2 in EDM2.
        c.f.  https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.VGG16_Weights
        preprocessing section

        Code copy/paste of ``__call__`` method in EDM2 dinov2, in particular
        ``edm2/calculate_metrics.py``
        """
        # XXX defer import
        from edm2.torch_utils import misc

        # Fast practical implementation that yields almost the same results.
        x = torch.nn.functional.interpolate(
            x.to(torch.float32), size=(224, 224), mode="bicubic", antialias=True
        )

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        return x


# --- copy/paste + modification of
# https://github.com/clovaai/generative-evaluation-prdc/blob/e320c1d2811d33081361a08f595b43830b78641c/prdc/prdc.py
"""
prdc
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import sklearn.metrics


def compute_pairwise_distance(data_x, data_y=None, *, n_jobs=8):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=n_jobs
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, *, n_jobs=8):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features, n_jobs=n_jobs)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, *, n_jobs=8):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print(
        "Num real: {} Num fake: {}".format(
            real_features.shape[0], fake_features.shape[0]
        )
    )

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k, n_jobs=n_jobs
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k, n_jobs=n_jobs
    )
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features, n_jobs=n_jobs
    )

    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
    ).mean()

    return dict(
        precision=float(precision),
        recall=float(recall),
        density=float(density),
        coverage=float(coverage),
    )
