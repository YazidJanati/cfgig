# Images

Dataset downloaded from https://image-net.org, section ILSVRC2012 validation set and test images

After downloading the data, the validation set contain 50k images (50 images per class).
A full description of the dataset can be found in ``ILSVRC2012_devkit_t12/data/imagenet-val/ILSVRC2012_devkit_t12/readme.txt`` in the section ``Validation images``.

To preprocess images, namely align them with [1], run the following
```bash
cd data
bash preprocess.sh
```

Note the script must be run from ``data`` folder.


# Audio LDM

The dataset can be downloaded from https://audiocaps.github.io .
The randomly selected prompts alongside their ids are available in ``test_random1000_48khz.json``


---

.. [1] Kynkäänniemi T, Aittala M, Karras T, Laine S, Aila T, Lehtinen J.
    Applying guidance in a limited interval improves sample and distribution quality in diffusion models.
    arXiv preprint arXiv:2404.07724. 2024 Apr 11.