from setuptools import setup


requirements = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers==4.48",
    "accelerate",
    "einops",
    "diffusers[torch]==0.32",
    "scikit-learn",
    "clip",
    "kornia",
    "numpy",
    "matplotlib",
    "omegaconf",
    "tqdm",
    "PyYAML",
    "pytorch-lightning",
    "taming-transformers-rom1504",
    "hydra-core",
    "pytorch_fid",
    "pandas",
    "joblib",
    "wandb",
    "torchaudio",
    # --- audio ldm
    "scikit-image",
    "scipy",
    "torchlibrosa",
    "ssr_eval",
    "resampy",
    "librosa",
]


setup(
    name="cfgig",
    version="0.0.0",
    install_requires=requirements,
)
