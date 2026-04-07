from pathlib import Path
from venv import logger

import pandas as pd
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "assignments" / "resources" / "happiness_project"
    output_dir = base_dir / "assignments_01" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("world_happiness_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []

    for file_path in csv_files:
        year = int(file_path.stem.split("_")[-1])

        with open(file_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline().strip()

        if ";" in first_line:
            sep = ";"
            decimal = ","
        else:
            sep = ","
            decimal = "."

        df = pd.read_csv(
            file_path,
            sep=sep,
            decimal=decimal,
            encoding="utf-8-sig"
        )

        df["year"] = year
        frames.append(df)

        logger.info(
            f"Loaded {file_path.name} with {len(df)} rows using sep='{sep}' and decimal='{decimal}'"
        )

    merged_df = pd.concat(frames, ignore_index=True)

    output_path = output_dir / "merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)

    logger.info(f"Merged dataset saved to {output_path}")
    logger.info(f"Merged dataset shape: {merged_df.shape}")

    return merged_df

@task
def compute_statistics(df):
    logger = get_run_logger()

    # unify happiness score
    df["happiness_score"] = df["Happiness score"].fillna(df["Ladder score"])

    # basic stats
    mean_val = df["happiness_score"].mean()
    median_val = df["happiness_score"].median()
    std_val = df["happiness_score"].std()

    logger.info(f"Mean happiness score: {mean_val}")
    logger.info(f"Median happiness score: {median_val}")
    logger.info(f"Std happiness score: {std_val}")

    # average by year
    yearly_avg = df.groupby("year")["happiness_score"].mean()
    logger.info(f"Average happiness by year:\n{yearly_avg}")

    # average by region
    region_avg = df.groupby("Regional indicator")["happiness_score"].mean()
    logger.info(f"Average happiness by region:\n{region_avg}")

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val
    }

@task
def create_visualizations(df):
    logger = get_run_logger()

    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "assignments_01" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    # Create unified happiness score
    df["happiness_score"] = df["Happiness score"].fillna(df["Ladder score"])

    # 1. Histogram of happiness scores
    plt.figure(figsize=(8, 5))
    plt.hist(df["happiness_score"].dropna(), bins=20)
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    histogram_path = output_dir / "happiness_histogram.png"
    plt.savefig(histogram_path)
    plt.close()
    logger.info(f"Saved histogram to {histogram_path}")

    # 2. Boxplot of happiness scores by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="year", y="happiness_score")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    boxplot_path = output_dir / "happiness_by_year.png"
    plt.savefig(boxplot_path)
    plt.close()
    logger.info(f"Saved boxplot by year to {boxplot_path}")

    # 3. Scatter plot of GDP per capita vs happiness score
    plt.figure(figsize=(8, 5))
    plt.scatter(df["GDP per capita"], df["happiness_score"])
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    scatter_path = output_dir / "gdp_vs_happiness.png"
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f"Saved GDP vs happiness scatter plot to {scatter_path}")

    # 4. Correlation heatmap
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True)
    plt.title("Correlation Heatmap")
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved correlation heatmap to {heatmap_path}")

@flow
def happiness_pipeline():
    logger = get_run_logger()
    logger.info("Pipeline started")

    merged_df = load_and_merge_data()

    stats = compute_statistics(merged_df)
    logger.info("Task 2 complete")

    create_visualizations(merged_df)
    logger.info("Task 3 complete")

if __name__ == "__main__":
    happiness_pipeline()