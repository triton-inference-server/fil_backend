# (DEPRECATED) Triton FIL backend with XGBoost

**THIS NOTEBOOK HAS BEEN DEPRECATED. FOR A SIMPLE AND CONCISE INTRODUCTION TO TRAINING AND DEPLOYING AN XGBOOST MODEL WITH THE FIL BACKEND, PLEASE SEE THE [CATEGORICAL FRAUD DETECTION](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks/categorical-fraud-detection) EXAMPLE NOTEBOOK.**

This notebook will eventually be reworked, split into smaller parts, and reintroduced for a later release. It is left here for historical reference, but some cells are known not to work with the latest versions of various Triton components.

This notebook is a reference for deploying an XGBoost model on Triton with the FIL backend. The notebook explains how one can deploy XGBoost model in Triton, check deployment status and send inference requests, set concurrent model execution and dynamic batching and find the best deployment configuration using Model Analyzer.

## Requirements
* NVIDIA GPU (Pascal+ required, recommended GPUs: T4, V100 or A100)
* [Latest NVIDIA driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* [Docker](https://docs.docker.com/get-docker/)
* [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Run the Triton Inference Server container

**Note:** Due to a bug in release 21.07, Triton's `model_analyzer` cannot be used with the FIL backend. If you wish to use the model analyzer, please use release 21.08 or later.

Before running the container, clone the repository and then run the container:

```
git clone https://github.com/triton-inference-server/fil_backend.git
cd fil_backend

docker run \
  -it \
  --gpus=all \
  --rm \
  --net=host \
  --name triton_fil \
  nvcr.io/nvidia/tritonserver:<tag>  # Put the appropriate tag here.
```

**Note:** The artifacts created by scripts inside the container are created with root permission. The user on host machine might not be able to modify the artifacts once the container exists. To avoid this issue, copy the notebook `docker cp simple_xgboost_example.ipynb <docker_ID>` and create the artifacts inside the container.

Now open up another terminal and copy the notebook from host into the container as follows:
```
docker cp notebooks/ triton_fil:/
```

## Starting Jupyter notebook
In the previous terminal perform the following steps:

### Install Jupyter notebook inside the Triton container
```
pip3 install jupyter
```
### Run Jupyter notebook inside the Triton container
Change directory to `/notebooks` folder and run the jupyter notebook:
```
cd /notebooks
jupyter notebook --allow-root --no-browser --port 7001
```

