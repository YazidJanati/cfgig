import os
import json
import hydra
from pathlib import Path
from datetime import datetime

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from local_paths import REPO_PATH
from metrics import Metric
from experiments_tools import update_sampler_config


@hydra.main(
    config_path=str(REPO_PATH / "configs/"),
    config_name="images",
)
def evaluation(config: DictConfig):

    update_sampler_config(config)

    print(f"Evaluating config {config.sampler.parameters}")

    imgs_save_dir = Path(config.eval.path_generated_data)

    # compute metrics (FID, FD_deno)
    metrics = {}
    for metric_name in config.eval.metrics:
        print(f"=========== {metric_name} ===========")
        fid_metric = Metric(
            metric=metric_name, batch_size=config.eval.batch_size, device="cuda:0"
        )
        val_metric = fid_metric.compute_FID(
            path_imgs=imgs_save_dir, seed=int(config.eval.seed)
        )
        metrics[metric_name] = val_metric

    print(f"=========== {'Metrics:'} ===========")
    print(metrics)


if __name__ == "__main__":
    evaluation()
