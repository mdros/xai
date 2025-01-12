import logging
import os
import hydra
import numpy as np
import pandas as pd
from time import time
import torch
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.utils

from counterfactuals.metrics import evaluate_cf_regression
from counterfactuals.cf_methods.regression_ppcef import PPCEFR
from counterfactuals.pipelines.nodes.helper_nodes import log_parameters, set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    run: neptune.Run,
    save_folder: str,
) -> torch.nn.Module:
    """
    Create a counterfactual model
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    X_test_origin = dataset.X_test
    y_test_origin = dataset.y_test

    logger.info("Creating counterfactual model")
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_loss)

    cf_method = PPCEFR(
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=disc_model_criterion,
        neptune_run=run,
    )

    logger.info("Calculating delta threshold")
    train_dataloader_for_delta = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    delta = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_delta),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    run["parameters/delta"] = delta
    logger.info(f"delta: {delta:.4f}")

    logger.info("Handling counterfactual generation")
    cf_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )

    time_start = time()
    x_cfs, x_origs, y_origs, y_cf_targets, logs = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        target_change=cfg.counterfactuals_params.target_change,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        alpha=cfg.counterfactuals_params.alpha,
        delta=delta,
    )

    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )

    pd.DataFrame(x_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    return x_cfs, x_origs, y_origs, y_cf_targets, delta


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: list,
    continuous_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    delta: float,
    run: neptune.Run,
    y_target: np.ndarray = None,
):
    """
    Calculate metrics for counterfactuals
    """
    logger.info("Calculating metrics")
    metrics = evaluate_cf_regression(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        median_log_prob=delta,
        y_target=y_target,
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
    return metrics


@hydra.main(config_path="./conf", config_name="ppcefr_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    log_parameters(cfg, run)
    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder, run)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
        dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

    gen_model = create_gen_model(cfg, dataset, gen_model_path, run)

    Xs_cfs, Xs, ys_orig, ys_target, delta = search_counterfactuals(
        cfg, dataset, gen_model, disc_model, run, save_folder
    )

    metrics = calculate_metrics(
        gen_model=gen_model,
        disc_model=disc_model,
        Xs_cfs=Xs_cfs,
        model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=Xs,
        y_test=ys_orig,
        y_target=ys_target,
        delta=delta,
        run=run,
    )
    print(metrics)
    run.stop()


if __name__ == "__main__":
    main()
