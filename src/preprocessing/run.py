#!/usr/bin/env python
"""
Preprocessing of the raw data downloaded from W&B
"""
import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="preprocessing")
    run.config.update(args)

    os.makedirs('data', exist_ok=True)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    raw_dataset_path = wandb.use_artifact(args.input_artifact).file()
    raw_data_df = pd.read_csv(raw_dataset_path)
    logger.info(f'there are {raw_data_df.shape[0]} records in the raw dataset')

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = raw_data_df['price'].between(min_price, max_price)
    prep_data_df = raw_data_df[idx].copy()
    # Drop latitude and longitude outside ranges
    idx = prep_data_df['longitude'].between(args.lowest_longitude, args.highest_longitude) \
        & prep_data_df['latitude'].between(args.lowest_latitude, args.highest_latitude)
    prep_data_df = prep_data_df[idx].copy()
    logger.info(f'there are {prep_data_df.shape[0]} records after preprocessing')
    # Convert last_review to datetime
    prep_data_df['last_review'] = pd.to_datetime(prep_data_df['last_review'])
    prep_data_path = os.path.join('data', args.output_artifact)
    prep_data_df.to_csv(prep_data_path, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        args.output_type,
        args.output_description,
    )
    artifact.add_file(prep_data_path)
    run.log_artifact(artifact)
    logger.info(f'artifact {args.output_artifact} uploaded on wandb')
    
    run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data preprocessing")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The full name of the W&B input raw artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The name to give to the W&B preprocessed artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="The type to set for the W&B preprocessed artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="The description to set for the W&B preprocessed artifact",
        required=True
    )

    parser.add_argument(
        "--lowest_latitude",
        type=float
    )
    parser.add_argument(
        "--highest_latitude",
        type=float
    )
    parser.add_argument(
        "--lowest_longitude",
        type=float
    )
    parser.add_argument(
        "--highest_longitude",
        type=float
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimum price to accept. Houses with lower price will be removed from dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price to accept. Houses with higher price will be removed from dataset",
        required=True
    )


    args = parser.parse_args()

    go(args)
