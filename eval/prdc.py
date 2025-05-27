import hydra
from pathlib import Path

from omegaconf import DictConfig
from local_paths import REPO_PATH
from metrics import PRDC


@hydra.main(config_path=str(REPO_PATH / "configs"), config_name="images")
def evaluation(config: DictConfig):

    imgs_save_dir = Path(config.eval.path_generated_data)

    # compute precision/recall density/coverage
    metrics = {}
    for latent_type in config.eval.prdc.latents_type:
        print(f"=========== {latent_type} ===========")

        fid_metric = PRDC(
            latent_type,
            n_neighbors=config.eval.prdc.n_neighbors,
            batch_size=config.eval.batch_size,
            n_jobs=4,
            device=config.device,
        )
        val_metric = [
            fid_metric.compute_prcd(
                path_imgs=imgs_save_dir, seed=int(config.eval.seed) + i
            )
            for i in range(config.eval.prdc.n_reps)
        ]

        metrics[latent_type] = val_metric

    print(f"=========== {'Metrics:'} ===========")
    print(metrics)


if __name__ == "__main__":
    evaluation()
