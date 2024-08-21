"""Script to split the data into training, validation and testing datasets."""
import os

import pandas as pd
import yaml


def main():
    """Split the data into training, validation and testing datasets. The
    validation dataset is for hyperparameter tuning and the testing dataset
    is for model evaluation at the end of the project."""
    with open('params.yaml', encoding="utf-8") as f:
        params = yaml.safe_load(f)

    input_file = params['load_data']['output_file']

    data = pd.read_parquet(input_file)

    train_split = params['data_split']['train_split']
    val_split = params['data_split']['val_split']
    test_split = params['data_split']['test_split']
    random_seed = params['random_seed']

    total_samples = len(data)
    train_samples = int(total_samples * train_split)
    val_samples = int(total_samples * val_split)
    test_samples = int(total_samples * test_split)

    train = data.sample(n=train_samples, random_state=random_seed)
    data = data.drop(train.index)
    valid = data.sample(n=val_samples, random_state=random_seed)
    data = data.drop(valid.index)
    test = data.sample(n=test_samples, random_state=random_seed)

    # Create data/processed directory if it does not exist
    os.makedirs("data/processed", exist_ok=True)

    print(f"Training dataset shape: {train.shape}")
    print(f"Validation dataset shape: {valid.shape}")
    print(f"Testing dataset shape: {test.shape}")

    train.to_parquet("data/processed/train.parquet")
    valid.to_parquet("data/processed/val.parquet")
    test.to_parquet("data/processed/test.parquet")


if __name__ == "__main__":
    main()
