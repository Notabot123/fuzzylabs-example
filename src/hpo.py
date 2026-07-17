"""Bayesian hyperparameter search (Optuna TPE sampler) over a small CNN config space.

Demo-scale by default: a handful of trials, each trained for a few epochs on
a subset of the training set, evaluated on the (full) validation set. This is
explicitly *not* meant to find the global optimum — it demonstrates the
pattern (define search space -> objective -> study) with knobs in
`src.config.HPOConfig` to scale up for a real search.
"""

import optuna
import torch
from torch.utils.data import DataLoader, Subset

from src.config import DEVICE, Config, ModelConfig
from src.train import train_model


def _subset_loader(dataset, size: int, batch_size: int, seed: int):
    size = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=True)


def make_objective(cfg: Config, train_dataset, val_loader, device=DEVICE):
    def objective(trial: optuna.Trial) -> float:
        model_cfg = ModelConfig(
            base_channels=trial.suggest_categorical(
                "base_channels", list(cfg.hpo.base_channels_choices)
            ),
            dropout=trial.suggest_float("dropout", *cfg.hpo.dropout_range),
            lr=trial.suggest_float("lr", *cfg.hpo.lr_range, log=True),
            weight_decay=trial.suggest_float("weight_decay", *cfg.hpo.weight_decay_range, log=True),
        )
        train_loader = _subset_loader(
            train_dataset, cfg.hpo.train_subset_size, cfg.data.batch_size, seed=cfg.data.seed
        )
        _, _, history = train_model(
            model_cfg, train_loader, val_loader, cfg.hpo.epochs_per_trial, device=device
        )
        return history[-1]["val_acc"]

    return objective


def run_hpo(cfg: Config, train_dataset, val_loader, device=DEVICE) -> tuple[ModelConfig, optuna.Study]:
    sampler = optuna.samplers.TPESampler(seed=cfg.data.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(cfg, train_dataset, val_loader, device),
        n_trials=cfg.hpo.n_trials,
        show_progress_bar=False,
    )
    best_cfg = ModelConfig(
        base_channels=study.best_params["base_channels"],
        dropout=study.best_params["dropout"],
        lr=study.best_params["lr"],
        weight_decay=study.best_params["weight_decay"],
    )
    return best_cfg, study
