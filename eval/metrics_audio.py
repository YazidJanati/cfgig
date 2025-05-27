import hydra

from omegaconf import DictConfig
from local_paths import REPO_PATH
from audioldm_eval import EvaluationHelper
from experiments_tools import update_sampler_config


@hydra.main(
    config_path=str(REPO_PATH / "configs"),
    config_name="audio",
)
def evaluation(config: DictConfig):

    update_sampler_config(config)

    eval_configs = config.eval[config.dataset]
    # compute metrics
    evaluator = EvaluationHelper(
        sampling_rate=config.sampling_rate,
        device=config.device,
        backbone=eval_configs.backbone,
    )
    metrics = evaluator.main(
        path_generated_data=str(config.eval.path_generated_data),
        path_ref_data=config.eval.path_ref_data,
        **eval_configs,
    )

    print(f"=========== {'Metrics:'} ===========")
    print(metrics)


if __name__ == "__main__":
    evaluation()
