Conditional Diffusion Models with Gibbs-like Guidance
=======================================================

Code base for the paper.
Among others, it introduces CFGIG, an algorithm for conditional diffusion models.
It generates high-fidelity samples without sacrificing the diverity of the conditional distribution.


## Setups the repository

Add the the following two files to

1. Install the code in editable mode.
```bash
pip install -e .
```
Details about the dependencies can be in ``setup.py``


2. then put the in ``src/local_paths.py`` file, the absolute paths to
    - the project
    - the folder where to put large files (such as data)

```python
from pathlib import Path

LARGE_FILE_DIR = Path("/path/large/files")  # <--- change it with the repository absolute path
REPO_PATH = Path("/path/to/repository")     # <--- change it with the absolute path of the folder
```


## Run experiments

To run experiments
```bash
# for images
python3 run_images.py

# for audios
python3 run_audios.py

# for the toy example [figure 2]
python3 toy_example.py
```
the behavior of the script can be adapted by changing the hyperparmeters in configuration files.
Namely, the global behavior of the scripts, such the model to run, the context, location where to save the output, ...
can be changes using the ``images.yaml`` and ``audios.yaml`` files.

Similarly, the hyperparameters of the algorithms/samplers are availble in ``configs/sampler`` and ``configs/base_sampler``, respectively and can also be changed.

- Available samplers: Heun, DDIM
- Available algorithm: CFG, CFG++, Limmited interval CFG (LI-CFG), CFGIG (ours)
- Available models:
    - Images
        - EDM2 model: their pseudos are available in ``src/edm2/__init__.py``
        - Stable Diffusion: "sdxl1.0", "sd2.1", "sd1.5"
    - Audio
        - audioldm2
        - audioldm2-large


## Evaluation

Use scripts in ``eval`` for evaluation
- ``fd.py`` to compute FID and FD_DINOv2
- ``prdc.py`` to compute Precision/Recall and Density/Coverage
- ``metrics_audios`` to compute FAD, KL and IS metrics

the behavior of these script can be changed by changing the the parameters the ``eval`` block in ``images.yaml`` and ``audios.yaml``.

Note that:
- **Computation of FD**: We use the pre-computed reference statistics of Imagenet provided in [EDM2](https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/) to compute FID and FD_DINOv2. These statistics should be put in the ``LARGE_FILE_DIR`` under the name ``img512.pkl``
- **Computation of Precision/Recall Density/Coverage**:  After downloading and prepocessing the data, it must be place in ``LARGE_FILE_DIR`` under the name ``imagenet512_val``
- **Computation of audio metrics**: the prompts used for evaluation are available in ``data/test_random1000_48khz.json``

In the README.md of the ``data`` folder, we provide details on downloading and preprocessing Imagenet validation set.
We also provide details on the AudioCaps test dataset and the randomly chosen prompts from the test set.


## Remarks

This code base rely on the code provided in the following repositories
- https://github.com/NVlabs/edm2
- https://github.com/haoheliu/AudioLDM2
- https://github.com/clovaai/generative-evaluation-prdc
