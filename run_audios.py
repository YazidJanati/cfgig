# %%
import time
import json
import hydra
from pathlib import Path
from omegaconf import DictConfig

import torch
import scipy.io.wavfile

from models import load_model
from samplers import ddim, heun
from algorithms import cfg_sampler, limitedcfg_sampler, cfgig_sampler, cfgpp_sampler
from utils import fix_seed
from experiments_tools import update_sampler_config
from metrics import get_gpu_memory_consumption
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
    config_name="audios",
)
def job_runner(config: DictConfig):

    device = config.device
    torch.set_default_device(device)
    seed = config.eval.seed
    fix_seed(seed)

    # prompts
    prompts_text = list(config.ctx)

    update_sampler_config(config)

    if config.save_dir is None:
        run_name = f"{config.sampler.name}-{config.base_sampler.name}"
        path_save_dir = REPO_PATH / "out-audios" / run_name
    else:
        path_save_dir = Path(config.save_dir_run)

    path_save_dir.mkdir(exist_ok=True, parents=True)

    print(f"Running config {config.sampler.parameters}\n")

    model = load_model(cond_model_id=config.model, device=device)

    initial_noise = torch.randn(
        (len(prompts_text) * config.eval.n_samples, *model.x_shape),
        device=device,
        dtype=model.dtype,
    )

    start_time = time.perf_counter()
    samples = samplers[config.sampler.name](
        initial_noise,
        denoiser_fn=model.denoiser_fn,
        ctx=prompts_text,
        mk_sigmas_fn=model.mk_sigmas_fn,
        sampler=base_samplers[config.base_sampler.name],
        **config.sampler.parameters,
        **config.base_sampler.parameters,
    )
    audios = model.decode(
        samples, prompts_text, select_best_audio=config.eval.select_best_audio
    )
    end_time = time.perf_counter()

    statistics_run = {
        "runtime": end_time - start_time,
        "gpu_consumption": get_gpu_memory_consumption(device),
    }
    print("Run statistics:")
    print(statistics_run)

    # case only best audios were selected
    # we endup with number of audios equal to number of prompts
    if config.eval.select_best_audio:
        for idx, audio in zip(prompts_text, audios):
            fname = path_save_dir / f"{idx}.wav"
            scipy.io.wavfile.write(
                filename=fname, rate=config.sampling_rate, data=audio
            )
    # usual case, we end-up with `n_prompts * n_samples` audios
    else:
        audios = audios.reshape(len(prompts_text), config.eval.n_samples, -1)
        for idx, audio in zip(prompts_text, audios):
            for i, audio_i in enumerate(audio):
                fname = path_save_dir / f"{idx}_{i}.wav"
                scipy.io.wavfile.write(
                    filename=fname, rate=config.sampling_rate, data=audio_i
                )

    print(f"[{device}] Finished.")


if __name__ == "__main__":
    job_runner()

# %%
