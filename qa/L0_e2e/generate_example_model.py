import argparse
import os

import cuml
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

CONFIG_TEMPLATE = ""

def generate_classification_data(classes=2, rows=1000, cols=32):
    """Generate classification training set"""
    with cuml.using_output_type('numpy'):
        data, labels = cuml.datasets.make_classification(
            n_samples=rows,
            n_features=cols,
            n_informative=cols // 3,
            n_classes=classes,
            random_state=0
        )
    return data, labels


def train_xgboost_classifier(data, labels, depth=25, trees=100):
    """Train XGBoost classification model"""
    if xgb is None:
        raise RuntimeError('XGBoost could not be imported')
    training_params = {
        'eval_metric': 'error',
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'max_depth': depth,
        'n_estimators': trees,
        'use_label_encoder': False,
        'predictor': 'gpu_predictor'
    }
    model = xgb.XGBClassifier(**training_params)

    return model.fit(data, labels)


def train_lightgbm_classifier(data, labels, depth=25, trees=100, classes=2):
    """Train LightGBM classification model"""
    if lgb is None:
        raise RuntimeError('LightGBM could not be imported')
    lgb_data = lgb.Dataset(data, label=labels)

    training_params = {
        'metric': 'binary_logloss',
        'objective': 'binary',
        'num_class': classes - 1,
        'max_depth': depth
    }
    model = lgb.train(training_params, lgb_data, trees)

    return model


def train_classifier(
        data,
        labels,
        model_type='xgboost',
        depth=25,
        trees=100,
        classes=2):
    """Train classification model"""
    if model_type == 'xgboost':
        return train_xgboost_classifier(
            data, labels, depth=depth, trees=trees
        )
    if model_type == 'lightgbm':
        return train_lightgbm_classifier(
            data, labels, depth=depth, trees=trees, classes=classes
        )

    raise RuntimeError('Unknown model type "{}"'.format(model_type))


def generate_regression_data(rows=1000, cols=32):
    with cuml.using_output_type('numpy'):
        data, labels = cuml.datasets.make_regression(
            n_samples=rows,
            n_features=cols,
            n_informative=cols // 3,
            random_state=0)
    return data, labels


def train_xgboost_regressor(data, targets, depth=25, trees=100):
    """Train XGBoost regression model"""

    if xgb is None:
        raise RuntimeError('XGBoost could not be imported')

    training_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'max_depth': depth,
        'n_estimators': trees,
        'predictor': 'gpu_predictor'
    }
    model = xgb.XGBRegressor(**training_params)

    return model.fit(data, targets)


def train_lightgbm_regressor(data, targets, depth=25, trees=100):
    """Train LightGBM regression model"""
    if lgb is None:
        raise RuntimeError('LightGBM could not be imported')
    lgb_data = lgb.Dataset(data, targets)

    training_params = {
        'metric': 'l2',
        'objective': 'regression',
        'max_depth': depth
    }
    model = lgb.train(training_params, lgb_data, trees)

    return model


def train_regressor(
        data,
        targets,
        model_type='xgboost',
        depth=25,
        trees=100):
    """Train regression model"""
    if model_type == 'xgboost':
        return train_xgboost_regressor(
            data, targets, depth=depth, trees=trees
        )
    if model_type == 'lightgbm':
        return train_lightgbm_regressor(
            data, targets, depth=depth, trees=trees
        )

    raise RuntimeError('Unknown model type "{}"'.format(model_type))

def generate_model(
        task='classification',
        model_type='xgboost',
        depth=25,
        trees=100,
        classes=2,
        samples=1000,
        features=32):
    """Generate a model with the given properties"""
    if task == 'classification':
        data, labels = generate_classification_data(
            classes=classes, rows=samples, cols=features
        )
        return train_classifier(
            data, labels, model_type=model_type, depth=depth, trees=trees,
            classes=classes
        )
    if task == 'regression':
        data, labels = generate_regression_data(
            rows=samples, cols=features
        )
        return train_regressor(
            data, labels, model_type=model_type, depth=depth, trees=trees
        )
    raise RuntimeError('Unknown model task "{}"'.format(task))


def serialize_model(model, directory, output_format='xgboost'):
    if output_format == 'xgboost':
        model_path = os.path.join(directory, 'xgboost.model')
        model.save_model(model_path)
        return model_path
    if output_format == 'xgboost_json':
        model_path = os.path.join(directory, 'xgboost.json')
        model.save_model(model_path)
        return model_path
    if output_format == 'lightgbm':
        model_path = os.path.join(directory, 'model.txt')
        model.save_model(model_path)
        return model_path
    raise RuntimeError(
        f'Unknown serialization format "{output_format}"'
    )

def generate_config(
        model_name,
        model_type='xgboost',
        predict_proba=False,
        task='classification',
        threshold=0.5,
        batching_window=30000):
    """Return a string with the full Triton config.pbtxt for this model
    """
    predict_proba = str(bool(predict_proba)).lower()
    output_class = str(task == 'classification').lower()

    return f""" name: "{model_name}"
backend: "fil"
max_batch_size: 8192
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 500 ]
  }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "{model_type}" }}
  }},
  {{
    key: "predict_proba"
    value: {{ string_value: "{predict_proba}" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "{output_class}" }}
  }},
  {{
    key: "threshold"
    value: {{ string_value: "{threshold}" }}
  }},
  {{
    key: "algo"
    value: {{ string_value: "ALGO_AUTO" }}
  }},
  {{
    key: "storage_type"
    value: {{ string_value: "AUTO" }}
  }},
  {{
    key: "blocks_per_sm"
    value: {{ string_value: "0" }}
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8, 16, 32, 64, 128, 1024, 2048, 4096, 8192]
  max_queue_delay_microseconds: {batching_window}
}}"""


def build_model(
        task='classification',
        model_type='xgboost',
        output_format=None,
        depth=25,
        trees=100,
        classes=2,
        samples=1000,
        features=32,
        model_repo=os.path.dirname(__file__),
        model_name=None,
        classification_threshold=0.5,
        predict_proba=False,
        batching_window=30000):
    """Train a model with given parameters, create a config file, and add it to
    the model repository"""

    if model_repo is None:
        model_repo = os.path.join(
            os.path.dirname(__file__),
            'model_repository'
        )

    if output_format is None:
        if model_type == 'xgboost':
            output_format = 'xgboost'
        elif model_type == 'lightgbm':
            output_format = 'lightgbm'
        else:
            raise RuntimeError('Unknown model type "{}"'.format(model_type))

    if (
        (
            model_type == 'xgboost' and
            output_format not in {'xgboost', 'xgboost_json'}
        ) or (
            model_type == 'lightgbm' and
            output_format not in {'lightgbm'}
        )
    ):
        raise RuntimeError(
            f'Output format "{output_format}" inconsistent with model type'
            f' "{model_type}"'
        )

    if model_name is None:
        model_name = f"{model_type}_{task}_{output_format}"

    config_dir = os.path.abspath(os.path.join(model_repo, model_name))
    model_dir = os.path.join(config_dir, '1')
    os.makedirs(model_dir, exist_ok=True)

    model = generate_model(
        task=task,
        model_type=model_type,
        depth=depth,
        trees=trees,
        classes=classes,
        samples=samples,
        features=features
    )

    serialize_model(model, model_dir, output_format=output_format)

    config = generate_config(
        model_name,
        model_type=model_type,
        predict_proba=predict_proba,
        task=task,
        threshold=classification_threshold,
        batching_window=batching_window
    )
    config_path = os.path.join(config_dir, 'config.pbtxt')

    with open(config_path, 'w') as config_file:
        config_file.write(config)

    return model_name


def parse_args():
    """Parse CLI arguments for model creation"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        choices=('lightgbm', 'xgboost'),
        default='xgboost',
        help='type of model',
    )
    parser.add_argument(
        '--task',
        choices=('classification', 'regression'),
        default='classification',
        help='whether model should perform classification or regression',
    )
    parser.add_argument(
        '--format',
        choices=('xgboost', 'xgboost_json', 'lightgbm'),
        default=None,
        help='serialization format for model',
    )
    parser.add_argument(
        '--depth',
        type=int,
        help='maximum model depth',
        default=25
    )
    parser.add_argument(
        '--trees',
        type=int,
        help='number of trees in model',
        default=100
    )
    parser.add_argument(
        '--classes',
        type=int,
        help='for classifiers, the number of classes',
        default=2
    )
    parser.add_argument(
        '--features',
        type=int,
        help='number of features in data',
        default=32
    )
    parser.add_argument(
        '--samples',
        type=int,
        help='number of training samples',
        default=1000
    )
    parser.add_argument(
        '--repo',
        help='path to model repository',
        default=None
    )
    parser.add_argument(
        '--name',
        help='name for model',
        default=None
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='for classifiers, the classification threshold',
        default=0.5
    )
    parser.add_argument(
        '--predict_proba',
        action='store_true',
        help='for classifiers, output class scores',
    )
    parser.add_argument(
        '--batching_window',
        type=int,
        help='window (in microseconds) for gathering batches'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(build_model(
        task=args.task,
        model_type=args.type,
        output_format=args.format,
        depth=args.depth,
        trees=args.trees,
        classes=args.classes,
        samples=args.samples,
        features=args.features,
        model_repo=args.repo,
        model_name=args.name,
        classification_threshold=args.threshold,
        predict_proba=args.predict_proba,
        batching_window=args.batching_window
    ))
