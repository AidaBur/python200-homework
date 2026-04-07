import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    """Convert a NumPy array to a pandas Series named 'values'."""
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    """Remove missing values from the Series."""
    return series.dropna()

@task
def summarize_data(series):
    """Return summary statistics for the cleaned Series."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow():
    """Run the Prefect pipeline."""
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary

if __name__ == "__main__":
    pipeline_flow()

# Why Prefect may be more overhead here:
# This pipeline is very small and only processes a handful of numbers.
# Plain Python functions are enough to handle this task, so using Prefect
# adds extra setup, logging, and orchestration that may not be necessary here.

# When Prefect could still be useful:
# Prefect would be more useful in real workflows that run on a schedule,
# need retries when a step fails, require logging and monitoring, or
# pull data from files, APIs, or databases. Even if each step stays simple,
# Prefect helps manage the workflow more reliably in production.