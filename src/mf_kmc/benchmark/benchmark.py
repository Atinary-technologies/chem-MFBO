"""Utilities to run the benchmarking strategies."""
import json
from pathlib import Path
from time import time
from typing import Any, Dict

import hydra
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

from mf_kmc import run
from mf_kmc.simulations import config


def save_results(
    results: Dict[str, Any], save_run_dir: Path, seed: int, steps: int, batch_size: int
) -> None:
    """Save results to dataframe for each seeded experiment."""

    run_dir = save_run_dir.joinpath(f"{seed}_{steps}steps_batch{batch_size}.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(run_dir)


@hydra.main(version_base=None, config_path="../../../config_bench", config_name="synthetic_benchmark")
def run_benchmark(cfg: DictConfig) -> None:
    """Run benchmark using Hydra's configuration file."""

    # get run dir for current experiment
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # get seeds
    seeds = list(range(cfg.n_repeats))

    t1 = time()

    # save simulation config and optimizer config
    config.init_config()

    sim_config = config.get_simulation_config(cfg.simulation)

    with open(run_dir.joinpath("sim_config.json"), "w") as f:
        sim_dict = sim_config.model_dump()
        json.dump(sim_dict, f, indent=4)

    # dump optim_config
    optim_config = config.get_optimizer_config(cfg.optimizer)

    with open(run_dir.joinpath("optim_config.json"), "w") as f:
        optim_dict = optim_config.model_dump()
        json.dump(optim_dict, f, indent=4)

    # run simulation
    if cfg.parallel:

        results_list = Parallel(n_jobs=-1)(
            delayed(run)(
                cfg.simulation,
                cfg.optimizer,
                budget=cfg.budget,
                cost_ratio=cfg.cost_ratio,
                init_samples_budget=cfg.init_samples_budget,
                low_fid_samps_ratio=cfg.low_fid_samps_ratio,
                batch_size=cfg.batch_size,
                lowfid_kernel=cfg.lowfid_kernel,
                bias_lowfid=cfg.bias_lowfid,
                multitask=cfg.multitask,
                seed=seed,
            )
            for seed in seeds
        )

        for result, seed in zip(results_list, seeds):
            save_results(result, run_dir, seed, cfg.budget, cfg.batch_size)

    else:
        # Running sequentially
        for seed in seeds:
            results = run(
                cfg.simulation,
                cfg.optimizer,
                budget=cfg.budget,
                cost_ratio=cfg.cost_ratio,
                init_samples_budget=cfg.init_samples_budget,
                low_fid_samps_ratio=cfg.low_fids_samps_ratio,
                batch_size=cfg.batch_size,
                lowfid_kernel=cfg.lowfid_kernel,
                bias_lowfid=cfg.bias_lowfid,
                multitask=cfg.multitask,
                seed=seed,
            )

            save_results(results, run_dir, seed, cfg.budget, cfg.batch_size)

    t2 = time()

    print(t2 - t1)


if __name__ == "__main__":

    run_benchmark()
