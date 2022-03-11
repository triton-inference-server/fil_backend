import os

def pytest_addoption(parser):
    default_repo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'model_repository'
    )
    parser.addoption(
        "--repo",
        action="store",
        default=default_repo_path
    )

    # option to skip treeshap tests for CPU only
    parser.addoption(
        "--no_shap",
        action="store_true"
    )
