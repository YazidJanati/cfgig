# Conditional Diffusion Models with Gibbs-like Guidance

Code base for the paper.  
Among others, it introduces **CFGIG**, an alternative to the vanilla Classifier-Free guidance algorithm for conditional diffusion models. It generates high-fidelity samples without sacrificing the **diversity** of the conditional distribution.

---

## Set up the repository

Add the following two files:

1. Install the code in editable mode:
   ```bash
   pip install -e .
   ```
   Details about the dependencies can be found in `setup.py`.

2. Then create the file `src/local_paths.py` with the absolute paths to:
   - the project
   - the folder where to store large files (such as data)

   ```python
   from pathlib import Path

   LARGE_FILE_DIR = Path("/path/large/files")  # <--- change to the path for large files
   REPO_PATH = Path("/path/to/repository")     # <--- change to the absolute path of the repository
   ```

---

## Run experiments

To run experiments:
```bash
# for images
python3 run_images.py

# for audio
python3 run_audios.py

# for the toy example [Figure 2]
python3 toy_example.py
```

The behavior of the scripts can be adjusted by changing the hyperparameters in the configuration files.  
In particular, global settings—such as the model to run, context, output directory, etc.—can be modified using the `images.yaml` and `audios.yaml` files.

Similarly, the hyperparameters of the algorithms/samplers are available in `configs/sampler` and `configs/base_sampler`, respectively.

- Available samplers: **Heun**, **DDIM**
- Available algorithms: **CFG**, **CFG++**, **Limited Interval CFG (LI-CFG)**, **CFGIG** (ours)
- Available models:
  - **Image**
    - **EDM2 model**: their pseudo-names are available in `src/edm2/__init__.py`
    - **Stable Diffusion**: `"sdxl1.0"`, `"sd2.1"`, `"sd1.5"`
  - **Audio**
    - `audioldm2`
    - `audioldm2-large`

---

## Evaluation

Use the scripts in the `eval` directory for evaluation:
- `fd.py`: compute FID and FD_DINOv2
- `prdc.py`: compute Precision/Recall and Density/Coverage
- `metrics_audios`: compute FAD, KL, and IS metrics

The behavior of these scripts can be modified by editing the `eval` block in `images.yaml` and `audios.yaml`.

**Notes:**
- **Computation of FD**: We use pre-computed ImageNet reference statistics from [EDM2](https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/) to compute FID and FD_DINOv2. These statistics should be placed in `LARGE_FILE_DIR` under the name `img512.pkl`.
- **Computation of Precision/Recall and Density/Coverage**: After downloading and preprocessing the data, place it in `LARGE_FILE_DIR` under the name `imagenet512_val`.
- **Computation of audio metrics**: Prompts used for evaluation are available in `data/test_random1000_48khz.json`.

Additional details for downloading and preprocessing the ImageNet validation set are provided in the `data/README.md` file.  
It also contains information on the AudioCaps test dataset and the selected evaluation prompts.

---

## Remarks

This code base relies on the following repositories:
- https://github.com/NVlabs/edm2
- https://github.com/haoheliu/AudioLDM2
- https://github.com/clovaai/generative-evaluation-prdc
