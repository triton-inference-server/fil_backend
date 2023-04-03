import os
from hypothesis import settings

settings.register_profile('dev', max_examples=10)
settings.register_profile('ci', max_examples=100)

def pytest_addoption(parser):
    default_repo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'model_repository'
    )
    default_cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '.model_cache'
    )
    parser.addoption(
        "--repo",
        action="store",
        default=default_repo_path
    )
    parser.addoption(
        "--model_cache_dir",
        action="store",
        default=default_cache_path
    )
