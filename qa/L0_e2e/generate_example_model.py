# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pickle
import numpy as np

import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.ensemble import RandomForestRegressor as cuRFR
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier as skRFC
    from sklearn.ensemble import RandomForestRegressor as skRFR
except ImportError:
    skRFC = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None


def generate_classification_data(classes=2, rows=1000, cols=32, cat_cols=0, seed=0, add_nans=False):
    """Generate classification training set"""
    if cat_cols > 0:
        output_type = 'cudf'
    else:
        output_type = 'numpy'

    with cuml.using_output_type(output_type):
        data, labels = cuml.datasets.make_classification(
            n_samples=rows,
            n_features=cols,
            n_informative=cols // 3,
            n_classes=classes,
            random_state=seed
        )
    if add_nans:
        if output_type == 'cudf':
            for i, col in enumerate(data.columns):
                data.loc[data.sample(frac=0.1, random_state=seed+i).index, col] = np.nan
            assert data.isnull().any().any()
        else:
            data.flat[np.random.RandomState(seed).choice(data.size, int(data.size * 0.5), replace=False)] = np.nan
            assert np.isnan(data).any()


    if cat_cols > 0:
        selected_cols = data.sample(n=min(cat_cols, cols), axis='columns', random_state=seed)
        data[selected_cols.columns].fillna(0, inplace=True)
        negatives = (selected_cols < 0)
        positives = (selected_cols >= 0)
        selected_cols = selected_cols.astype('object')
        selected_cols[negatives] = 0
        selected_cols[positives] = 1
        data[selected_cols.columns] = selected_cols.astype('category')
        data = data.to_pandas()
        labels = labels.to_pandas()
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
        'predictor': 'gpu_predictor',
        'enable_categorical':True
    }
    model = xgb.XGBClassifier(**training_params)

    return model.fit(data, labels)


def train_lightgbm_classifier(data, labels, depth=25, trees=100, classes=2):
    """Train LightGBM classification model"""
    if lgb is None:
        raise RuntimeError('LightGBM could not be imported')
    
    training_params = {
        'max_depth': depth,
        'num_estimators':trees,
        'verbose': -1
    }
    return lgb.LGBMClassifier(**training_params).fit(data,labels)


def train_lightgbm_rf_classifier(data, labels, depth=25, trees=100, classes=2):
    """Train LightGBM classification model"""
    if lgb is None:
        raise RuntimeError('LightGBM could not be imported')
    lgb_data = lgb.Dataset(data, label=labels)

    if classes <= 2:
        classes = 1
        objective = 'binary'
        metric = 'binary_logloss'
    else:
        objective = 'multiclass'
        metric = 'multi_logloss'

    training_params = {
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'boosting': 'rf',
        'metric': metric,
        'objective': objective,
        'num_class': classes,
        'max_depth': depth,
        'verbose': -1
    }
    model = lgb.train(training_params, lgb_data, trees)

    return model


def train_sklearn_classifier(data, labels, depth=25, trees=100):
    """Train SKLearn classification model"""
    if skRFC is None:
        raise RuntimeError('SKLearn could not be imported')
    model = skRFC(
        max_depth=depth, n_estimators=trees, random_state=0
    )

    return model.fit(data, labels)

def train_sklearn_gbm_classifier(data, labels, depth=25, trees=100):
    if sklearn is None:
        raise RuntimeError('SKLearn could not be imported')
    model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=trees,max_depth=depth, init="zero")
    return model.fit(data, labels)


def train_cuml_classifier(data, labels, depth=25, trees=100):
    """Train SKLearn classification model"""
    model = cuRFC(
        max_depth=depth, n_estimators=trees, random_state=0
    )

    return model.fit(data, labels)


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
    if model_type == 'lightgbm_rf':
        return train_lightgbm_rf_classifier(
            data, labels, depth=depth, trees=trees, classes=classes
        )
    if model_type == 'cuml':
        return train_cuml_classifier(
            data, labels, depth=depth, trees=trees
        )
    if model_type == 'sklearn':
        return train_sklearn_classifier(
            data, labels, depth=depth, trees=trees
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
    """Train XGBoost regresscion model"""

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

    training_params = {
        'num_estimators': trees,
        'max_depth': depth,
        'verbose': -1
    }

    return lgb.LGBMRegressor(**training_params).fit(data,targets)


def train_lightgbm_rf_regressor(data, targets, depth=25, trees=100):
    """Train LightGBM regression model"""
    if lgb is None:
        raise RuntimeError('LightGBM could not be imported')
    lgb_data = lgb.Dataset(data, targets)

    training_params = {
        'boosting': 'rf',
        'metric': 'l2',
        'objective': 'regression',
        'max_depth': depth,
        'verbose': -1
    }
    model = lgb.train(training_params, lgb_data, trees)

    return model


def train_sklearn_regressor(data, targets, depth=25, trees=100):
    """Train SKLearn regression model"""
    if skRFR is None:
        raise RuntimeError('SKLearn could not be imported')
    model = skRFR(
        max_depth=depth, n_estimators=trees, random_state=0
    )

    return model.fit(data, targets)


def train_sklearn_gbm_regressor(data, targets, depth=25, trees=100):
    model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=trees,max_depth=depth, init="zero")
    return model.fit(data, targets)

def train_cuml_regressor(data, targets, depth=25, trees=100):
    """Train cuML regression model"""
    model = cuRFR(
        max_depth=depth, n_estimators=trees, random_state=0
    )

    return model.fit(data, targets)


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
    if model_type == 'lightgbm_rf':
        return train_lightgbm_rf_regressor(
            data, labels, depth=depth, trees=trees
        )
    if model_type == 'sklearn':
        return train_sklearn_regressor(
            data, targets, depth=depth, trees=trees
        )
    if model_type == 'cuml':
        return train_cuml_regressor(
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
        features=32,
        cat_features=0):
    """Generate a model with the given properties"""
    if cat_features != 0 and model_type != 'lightgbm':
        raise NotImplementedError(
            'Categorical feature generation has not yet been implemented for'
            ' non-LightGBM models'
        )
    if task == 'classification':
        data, labels = generate_classification_data(
            classes=classes, rows=samples, cols=features, cat_cols=cat_features
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
        model.booster_.save_model(model_path)
        return model_path
    if output_format == 'pickle':
        model_path = os.path.join(directory, 'model.pkl')
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        return model_path
    raise RuntimeError(
        f'Unknown serialization format "{output_format}"'
    )


def generate_config(
        model_name,
        model_repo,
        *,
        instance_kind='gpu',
        model_format='xgboost',
        features=32,
        num_classes=2,
        predict_proba=False,
        use_experimental_optimizations=True,
        task='classification',
        threshold=0.5,
        max_batch_size=8192,
        storage_type="AUTO",
        generate_shap=False):
    if predict_proba:
        output_dim = num_classes
    else:
        output_dim = 1
    predict_proba = str(bool(predict_proba)).lower()
    use_experimental_optimizations = str(bool(use_experimental_optimizations)).lower()
    output_class = str(task == 'classification').lower()

    if model_format == 'pickle':
        model_format = 'treelite_checkpoint'

    # Add treeshap output to xgboost_shap model
    treeshap_output_dim = num_classes if num_classes > 2 or 'cuml' in model_name else 1
    if treeshap_output_dim == 1:
        treeshap_output_str = f"{features + 1}"
    else:
        treeshap_output_str = f"{treeshap_output_dim}, {features + 1}"
    treeshap_output = ""
    if generate_shap:
        treeshap_output = f"""
        ,{{
            name: "treeshap_output"
            data_type: TYPE_FP32
            dims: [ {treeshap_output_str} ]
        }}
        """

    config = f"""name: "{model_name}"
backend: "fil"
max_batch_size: {max_batch_size}
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {features} ]
  }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {output_dim} ]
  }}
 {treeshap_output}
]
instance_group [{{ kind: {instance_kind} }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "{model_format}" }}
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
    value: {{ string_value: "{storage_type}" }}
  }},
  {{
    key: "blocks_per_sm"
    value: {{ string_value: "0" }}
  }},
  {{
    key: "use_experimental_optimizations"
    value: {{ string_value: "{use_experimental_optimizations}" }}
  }}
]

dynamic_batching {{ }}"""
    config_dir = os.path.abspath(os.path.join(model_repo, model_name))
    config_path = os.path.join(config_dir, 'config.pbtxt')
    with open(config_path, 'w') as config_file:
        config_file.write(config)



def build_model(
        task='classification',
        model_type='xgboost',
        instance_kind='gpu',
        output_format=None,
        depth=25,
        trees=100,
        classes=2,
        samples=1000,
        features=32,
        cat_features=0,
        model_repo=os.path.dirname(__file__),
        model_name=None,
        classification_threshold=0.5,
        predict_proba=False,
        use_experimental_optimizations=True,
        max_batch_size=8192,
        storage_type="AUTO"):
    """Train a model with given parameters, create a config file, and add it to
    the model repository"""

    if instance_kind == 'gpu':
        instance_kind = 'KIND_GPU'
    elif instance_kind == 'cpu':
        instance_kind = 'KIND_CPU'
    else:
        raise ValueError("instance_kind must be either 'gpu' or 'cpu'")

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
        elif model_type in {'sklearn', 'cuml'}:
            output_format = 'pickle'
        else:
            raise RuntimeError('Unknown model type "{}"'.format(model_type))

    if (
        (
            model_type == 'xgboost' and
            output_format not in {'xgboost', 'xgboost_json'}
        ) or (
            model_type == 'lightgbm' and
            output_format not in {'lightgbm'}
        ) or (
            model_type == 'sklearn' and
            output_format not in {'pickle'}
        ) or (
            model_type == 'cuml' and
            output_format not in {'pickle'}
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
        features=features,
        cat_features=cat_features
    )

    serialize_model(model, model_dir, output_format=output_format)

    generate_config(
        model_name,
        model_repo,
        instance_kind=instance_kind,
        model_format=output_format,
        features=features,
        num_classes=classes,
        predict_proba=predict_proba,
        use_experimental_optimizations=use_experimental_optimizations,
        task=task,
        threshold=classification_threshold,
        max_batch_size=max_batch_size,
        storage_type=storage_type,
        generate_shap='shap' in model_name
    )

    return model_name


def parse_args():
    """Parse CLI arguments for model creation"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        choices=('lightgbm', 'lightgbm_rf', 'xgboost', 'sklearn', 'cuml'),
        default='xgboost',
        help='type of model',
    )
    parser.add_argument(
        '--instance_kind',
        choices=('gpu', 'cpu'),
        default='gpu',
        help='Whether to use GPU or CPU for prediction',
    )
    parser.add_argument(
        '--task',
        choices=('classification', 'regression'),
        default='classification',
        help='whether model should perform classification or regression',
    )
    parser.add_argument(
        '--format',
        choices=('xgboost', 'xgboost_json', 'lightgbm', 'pickle'),
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
        '--cat_features',
        type=int,
        help='number of categorical features (must be <= features)',
        default=0
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
        '--disable_experimental_optimizations',
        action='store_true',
        help='for classifiers, output class scores',
    )
    parser.add_argument(
        '--max_batch_size',
        type=int,
        help='largest batch size allowed for this model',
        default=8192
    )
    parser.add_argument(
        '--storage_type',
        choices=['AUTO', 'DENSE', 'SPARSE', 'SPARSE8'],
        help='storage type used to load this model in FIL',
        default='AUTO'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(build_model(
        task=args.task,
        model_type=args.type,
        instance_kind=args.instance_kind,
        output_format=args.format,
        depth=args.depth,
        trees=args.trees,
        classes=args.classes,
        samples=args.samples,
        features=args.features,
        cat_features=args.cat_features,
        model_repo=args.repo,
        model_name=args.name,
        classification_threshold=args.threshold,
        predict_proba=args.predict_proba,
        use_experimental_optimizations=(
            not args.disable_experimental_optimizations
        ),
        max_batch_size=args.max_batch_size,
        storage_type=args.storage_type
    ))
