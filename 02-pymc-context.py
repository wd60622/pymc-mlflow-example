"""Using for a PyMC context.

What would someone be interested in logging?

- Information about the model
- Information about the fit
- etc

"""

import mlflow

import os

from utils import (
    mlflow_set_tracking_uri,
    define_normal_model,
    generate_normal_data,
)

import arviz as az
import pymc as pm
import numpy as np


seed = sum(map(ord, "Logging PyMC model"))
rng = np.random.default_rng(seed)


n = 100
data = generate_normal_data(n=n, rng=rng, mu=2.5, sigma=3.5)

# MLflow setup
mlflow_set_tracking_uri()

# Point to MLflow experiment
experiment_name = "02-pymc-context"
mlflow.set_experiment(experiment_name)

# Start a run
mlflow.start_run()

tune = 500
draws = 1000
chains = 2
nuts_sampler = "numpyro"
mlflow.log_param("tune", tune)
mlflow.log_param("draws", draws)
mlflow.log_param("chains", chains)
mlflow.log_param("nuts_sampler", nuts_sampler)

with define_normal_model(data=data) as model:
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        nuts_sampler=nuts_sampler,
    )

    # Log various artifacts associated with the model fit
    df_summary = az.summary(idata)
    mlflow.log_metric("max_r_hat", df_summary["r_hat"].max())

    summary_file = "summary.csv"
    df_summary.reset_index().to_csv(summary_file)
    mlflow.log_artifact(summary_file)
    os.remove(summary_file)

    mlflow.log_text(
        model.str_repr(),
        "model.txt",
    )


mlflow.end_run()
