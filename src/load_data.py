"""Script to load data from the Scikit-learn dataset and save it as a
parquet file."""
import os

import pandas as pd
import yaml
from sklearn.datasets import load_diabetes


def main():
    """Load the diabetes dataset and save it as a parquet file."""
    with open('params.yaml', encoding="utf-8") as f:
        params = yaml.safe_load(f)

    diabetes = load_diabetes(as_frame=True)
    # Create data/raw directory if it does not exist
    os.makedirs("data/raw", exist_ok=True)
    data_inputs = diabetes.data
    data_target = diabetes.target
    data = pd.concat([data_inputs, data_target], axis=1)

    output_file = params['load_data']['output_file']
    data.to_parquet(output_file)


if __name__ == "__main__":
    main()
