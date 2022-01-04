import logging

import pandas as pd
import numpy as np
import scipy.stats

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def test_column_names(data: pd.DataFrame, ref_data: pd.DataFrame):

    these_columns = list(data.columns.values)

    ref_columns = list(ref_data.columns.values)

    # raise error if new dataset doesn't have all reference columns
    for col in ref_columns:
        assert col in these_columns, f'column {col} not found in the new data'

    # give a warning if the new dataset has an unknown column
    for col in these_columns:
        if not col in ref_columns:
            logger.warning(f'found new column {col} in the new dataset')


def test_neighborhood_names(data: pd.DataFrame, ref_data: pd.DataFrame):

    known_neighs = list(ref_data['neighbourhood_group'].unique())

    neighs = list(data['neighbourhood_group'].unique())

    # raise error if new dataset has an unknown neighborhood group
    for neigh in neighs:
        assert neigh in known_neighs, f'neighborhood group "{neigh}" unknown'


def test_proper_boundaries(
    data: pd.DataFrame, 
    lowest_latitude: float,
    highest_latitude: float,
    lowest_longitude: float,
    highest_longitude: float,
):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(lowest_longitude, highest_longitude) \
        & data['latitude'].between(lowest_latitude, highest_latitude)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data: pd.DataFrame, min_row_count: float, max_row_count: float):
    """
    Check that the dataset is not too small or too large
    """
    row_count = data.shape[0]

    assert min_row_count < row_count < max_row_count, 'the dataset is too large or too small'


def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    Check that the dataset is not too small or too large
    """

    assert data['price'].between(min_price, max_price).all(), 'there are rows with price outside the defined boundaries'
