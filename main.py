import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_default_steps = [
    "download",
    "preprocessing",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _default_steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": config["etl"]["raw_artifact_name"],
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "eda" in active_steps:
            # Start jupyter notebook for EDA analysis
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'eda'),
                "main",
                parameters={
                    "input_artifact": config["etl"]["raw_artifact_name"] + ":latest"
                },
            )

        if "preprocessing" in active_steps:
            # Preprocess the raw data and upload the updated data to W&B
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'preprocessing'),
                "main",
                parameters={
                    "input_artifact": config["etl"]["raw_artifact_name"] + ":latest",
                    "output_artifact": config["etl"]["preprocessed_artifact_name"],
                    "output_type": "preprocessed_data",
                    "output_description": "Input data after preprocessing",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            # Test the data over the reference dataset
            _ = mlflow.run(
                os.path.join(root_path, 'src', 'data_check'),
                "main",
                parameters={
                    "csv": config["etl"]["preprocessed_artifact_name"] + ":latest",
                    "ref": config["etl"]["preprocessed_artifact_name"] + ":reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "lowest_latitude": config["data_check"]["boundaries"]["lowest_latitude"],
                    "highest_latitude": config["data_check"]["boundaries"]["highest_latitude"],
                    "lowest_longitude": config["data_check"]["boundaries"]["lowest_longitude"],
                    "highest_longitude": config["data_check"]["boundaries"]["highest_longitude"],
                    "min_row_count": config["data_check"]["min_row_count"],
                    "max_row_count": config["data_check"]["max_row_count"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            # Split the data in train and test datasets
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input": config["etl"]["preprocessed_artifact_name"] + ":latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
