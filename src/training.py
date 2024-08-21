"""Script to create a RandomForestRegressor model and save it to disk."""
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml


def main():
    """Load the training data and fit a RandomForestRegressor model."""
    with open('params.yaml', encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_file = params['data_split']['train_data_path']

    train = pd.read_parquet(train_file)
    x_train = train.drop("target", axis=1)
    y_train = train["target"]

    random_seed = params['random_seed']
    training_params = params['training']
    model_file = training_params['model_file']
    criterion = training_params['criterion']
    n_estimators = training_params['n_estimators']
    max_depth = training_params['max_depth']
    min_samples_leaf = training_params['min_samples_leaf']
    max_features = training_params['max_features']

    rf = RandomForestRegressor(
        criterion=criterion, n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        random_state=random_seed)
    rf.fit(x_train, y_train)

    # Create models directory if it does not exist
    os.makedirs('models', exist_ok=True)

    with open(model_file, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    main()
