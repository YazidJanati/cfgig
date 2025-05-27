import os
import time
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from models import load_model
from samplers import ddim, heun
from algorithms import cfg_sampler, limitedcfg_sampler, cfgig_sampler, cfgpp_sampler
from utils import get_pil_image, fix_seed
from metrics import get_gpu_memory_consumption
from experiments_tools import update_sampler_config
from local_paths import REPO_PATH


base_samplers = {
    "ddim": ddim,
    "heun": heun,
}

samplers = {
    "cfg": cfg_sampler,
    "cfgpp": cfgpp_sampler,
    "limited_cfg": limitedcfg_sampler,
    "cfgig": cfgig_sampler,
}


@hydra.main(
    config_path=str(REPO_PATH / "configs"),
    config_name="images",
)
def job_runner(config: DictConfig):

    device = config.device
    torch.set_default_device(device)

    update_sampler_config(config)

    print(f"[{device}] Starting ...")
    print(f"Running config {config.sampler.parameters}\n")

    print(os.getcwd())

    if config.save_dir is None:
        run_name = f"{config.sampler.name}-{config.base_sampler.name}"
        path_save_dir = REPO_PATH / "out-images" / run_name
    else:
        path_save_dir = Path(config.save_dir_run)

    path_save_dir.mkdir(exist_ok=True, parents=True)

    fix_seed(config.seed)
    print("\n Loading model...\n")
    model = load_model(cond_model_id=config.model, device=device)

    ctx = list(config.ctx)
    initial_noise = torch.randn(
        (len(ctx) * config.n_samples, *model.x_shape), device=device
    )

    start_time = time.perf_counter()
    samples = samplers[config.sampler.name](
        initial_noise,
        denoiser_fn=model.denoiser_fn,
        ctx=ctx,
        mk_sigmas_fn=model.mk_sigmas_fn,
        sampler=base_samplers[config.base_sampler.name],
        **config.sampler.parameters,
        **config.base_sampler.parameters,
    )
    samples = model.decode(samples)
    end_time = time.perf_counter()

    ims = get_pil_image(samples)

    statistics_run = {
        "runtime": end_time - start_time,
        "gpu_consumption": get_gpu_memory_consumption(device),
    }
    print("Run statistics:")
    print(statistics_run)

    # save images
    ctx = np.array(ctx).repeat(config.n_samples)
    for idx, (ctx_val, im) in enumerate(zip(ctx, ims)):

        imgs_save_dir = path_save_dir / str(ctx_val.item())
        imgs_save_dir.mkdir(exist_ok=True)

        im.save(imgs_save_dir / f"{idx % config.n_samples}.jpeg")

    print(f"[{device}] Finished.")


if __name__ == "__main__":
    job_runner()
