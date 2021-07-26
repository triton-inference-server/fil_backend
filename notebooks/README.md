# Trtion FIL backend with XGBoost

This notebook is a reference for deploying a forest model trained using XGBoost library in Triton Inference Server with Forest Inference Library (FIL) backend. The notebook explains how one can deploy XGBoost model in Triton, check deployment status and send inference requests, set concurrent model execution and dynamic batching and find the best deployment configuration using Model Analyzer.

## Requirements
* Nvidia GPU (Pascal+ Recommended GPUs: T4, V100 or A100)
* [Latest NVIDIA driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* [Docker](https://docs.docker.com/get-docker/)
* [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Run the Triton Inference Server container 

**Note:** Please build the container locally inorder to use `model_analyzer` until the 21.08 release.

To run this notebook, pull the Triton container or build the container locally as mentioned in the README of [Triton Inference Server FIL backend](https://github.com/triton-inference-server/fil_backend#pre-built-container) Github repo. Before running the container, clone the repository and then run the container:

```
git clone https://github.com/triton-inference-server/fil_backend.git
cd fil_backend/notebooks

docker run \
  -it \
  --gpus=all \
  --rm \
  --net=host \
  -v $PWD:/notebook \                    
  nvcr.io/nvidia/tritonserver:<tag>  # Put the appropriate tag here.  
```
**Note:** The artifacts created by scripts inside the container are created with root permission. The user on host machine might not be able to modify the artifacts once the container exists. To avoid this issue, either make sure you have `sudo` access on host machine or copy the notebook `docker cp simple_xgboost_example.ipynb <docker_ID>` and create the artifacts inside the container.

## Install Jupyter notebook inside the Triton container
```
pip3 install jupyter
```

## Starting Jupyter notebook
Change directory to `/notebook` folder and run the jupyter notebook:
```
cd /notebook
jupyter notebook --allow-root --no-browser --port 7001
```

