"""Stage to evaluate the model on the validation data. Once the model is
selected and its hyperparameters are tuned, it will be evaluated on the
test data at the end of the pipeline."""
import pickle

import pandas as pd
import yaml
from dvclive import Live
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    """Calculate evaluation metrics on the validation data."""
    with open('params.yaml', encoding="utf-8") as f:
        params = yaml.safe_load(f)

    validation_file = params['data_split']['val_data_path']
    model_file = params['training']['model_file']

    validation = pd.read_parquet(validation_file)
    x_val = validation.drop("target", axis=1)
    y_val = validation["target"]

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(x_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # Initialize dvclive
    live = Live("eval")
    live.log_metric("mse", mse)
    live.log_metric("r2", r2)
    live.log_metric("mae", mae)

    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Feature Importance")
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=x_val.columns)
    forest_importances = forest_importances.sort_values(ascending=False)
    forest_importances.plot.bar(ax=axes)
    live.log_image("feature_importance.png", fig)
    live.next_step()

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")


if __name__ == '__main__':
    main()
