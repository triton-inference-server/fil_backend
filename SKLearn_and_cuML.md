# Scikit-Learn and cuML random forest support

## Model Serialization

While LightGBM and XGBoost have their own serialization formats that are
directly supported by the Triton FIL backend, random forest models trained with
[Scikit-Learn](https://scikit-learn.org/stable/modules/model_persistence.html)
or [cuML](https://docs.rapids.ai/api/cuml/stable/pickling_cuml_models.html) are
generally serialized using Python's
[pickle](https://docs.python.org/3/library/pickle.html) module. In order to
avoid a round-trip through Python in Triton, the FIL backend instead requires
that these pickled models first be converted to Treelite's binary checkpoint
format. Note that this also allows you to make use of *any* Treelite-supported
model framework in Triton simply by exporting to the binary checkpoint format.

The FIL backend repo includes scripts for easy conversion from
pickle-serialized cuML or Scikit-Learn models to Treelite checkpoints. You can
download the relevant script for Scikit-Learn
[here](https://raw.githubusercontent.com/triton-inference-server/fil_backend/main/scripts/convert_sklearn)
and for cuML
[here](https://raw.githubusercontent.com/triton-inference-server/fil_backend/main/scripts/convert_cuml.py).

## Prerequisites

To use the Scikit-Learn conversion script, you must run it from within a Python
environment containing both
[Scikit-Learn](https://scikit-learn.org/stable/install.html) and
[Treelite](https://treelite.readthedocs.io/en/latest/install.html). To use the
cuML conversion script, you must run it from within a Python environment
containing [cuML](https://rapids.ai/start.html).

For convenience, a conda environment config file
[is provided](https://raw.githubusercontent.com/triton-inference-server/fil_backend/main/scripts/environment.yml)
which will install all three of these prerequisites:

```
conda env update -f scripts/environment.yml
conda activate triton_scripts
```

## Converting to Treelite checkpoints

**NOTE:** The following steps are **not** necessary for LightGBM or XGBoost
models.  The FIL backend supports the native serialization formats for these
frameworks directly.

If you already have a Scikit-Learn or cuML RF model saved as a pickle file
(`model.pkl`), place it in a directory structure as follows:

```
model_repository/
`-- fil
    |-- 1
    |   `-- model.pkl
    `-- config.pbtxt
```

Then perform the conversion by running either:
```bash
./convert_sklearn model_repository/fil/1/model.pkl
```
for Scikit-Learn models or
```bash
./convert_cuml.py model_repository/fil/1/model.pkl
```
for cuML models. This will generate a `checkpoint.tl` file in the model
repository in the necessary location. You can then proceed as with any other
model type, setting the `model_type` parameter in `config.pbtxt` to
`"treelite_checkpoint"`.

Note that Treelite does not guarantee compatibility between minor release
versions for its binary checkpoint model, so it is recommended that you keep
the original pickle file. If you later make use of a newer version of Treelite,
you can simple re-run the conversion on this pickle file.

