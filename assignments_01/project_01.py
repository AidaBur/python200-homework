from pathlib import Path

import pandas as pd
from prefect import task, flow, get_run_logger


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


@flow
def happiness_pipeline():
    logger = get_run_logger()
    logger.info("Pipeline started")

    merged_df = load_and_merge_data()
    logger.info(f"Task 1 complete. Total rows loaded: {len(merged_df)}")


if __name__ == "__main__":
    happiness_pipeline()