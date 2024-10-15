#!/usr/bin/env python3
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


"""sklearn RF/GBDT to Treelite checkpoint converter

Given a path to a pickle file containing a scikit-learn random forest (or
gradient boosting) model, this script will generate a Treelite checkpoint file
representation of the model in the same directory.
"""

import argparse
import pathlib
import pickle

import treelite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_file", help="Path to the pickle file to convert")
    args = parser.parse_args()

    with open(args.pickle_file, "rb") as f:
        model = pickle.load(f)

    model_dir = pathlib.Path(args.pickle_file).resolve().parent
    out_path = model_dir / "checkpoint.tl"

    tl_model = treelite.sklearn.import_model(model)
    tl_model.serialize(out_path)
