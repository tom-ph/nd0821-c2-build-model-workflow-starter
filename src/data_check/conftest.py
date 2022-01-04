import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--lowest_latitude", action="store")
    parser.addoption("--highest_latitude", action="store")
    parser.addoption("--lowest_longitude", action="store")
    parser.addoption("--highest_longitude", action="store")
    parser.addoption("--min_row_count", action="store")
    parser.addoption("--max_row_count", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def lowest_latitude(request):
    lowest_latitude = request.config.option.lowest_latitude

    if lowest_latitude is None:
        pytest.fail("You must provide latitude and longitude boundaries")

    return float(lowest_latitude)


@pytest.fixture(scope='session')
def highest_latitude(request):
    highest_latitude = request.config.option.highest_latitude

    if highest_latitude is None:
        pytest.fail("You must provide latitude and longitude boundaries")

    return float(highest_latitude)

    
@pytest.fixture(scope='session')
def lowest_longitude(request):
    lowest_longitude = request.config.option.lowest_longitude

    if lowest_longitude is None:
        pytest.fail("You must provide latitude and longitude boundaries")

    return float(lowest_longitude)


@pytest.fixture(scope='session')
def highest_longitude(request):
    highest_longitude = request.config.option.highest_longitude

    if highest_longitude is None:
        pytest.fail("You must provide latitude and longitude boundaries")

    return float(highest_longitude)


@pytest.fixture(scope='session')
def min_row_count(request):
    min_row_count = request.config.option.min_row_count

    if min_price is None:
        pytest.fail("You must provide min_row_count")

    return float(min_row_count)


@pytest.fixture(scope='session')
def max_row_count(request):
    max_row_count = request.config.option.max_row_count

    if max_row_count is None:
        pytest.fail("You must provide max_row_count")

    return float(max_row_count)


@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
