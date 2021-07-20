# Trtion FIL backend with XGBoost

This notebook is a reference for deploying a forest model trained using XGBoost library in Triton Inference Server with Forest Inference Library (FIL) backend. The notebook explains how one can deploy XGBoost model in Triton, check deployment status and send inference requests, set concurrent model execution and dynamic batching and find the best deployment configuration using Model Analyzer.

## Requirements
* Nvidia GPU (T4 or V100 or A100)
* [Latest NVIDIA driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* [Docker](https://docs.docker.com/get-docker/)
* [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Run the Triton Inference Server container 

To run this notebook, pull the Triton container as mentioned in the README of [Triton Inference Server FIL backend](https://github.com/triton-inference-server/fil_backend) Github repo. Before running the container, create a directory to store model and other artifacts called `/data` and then run the container with the following command:

```
docker run \
  - it \
  --gpus=all \
  --rm \
  --net=host \
  -v /data:/data \
  nvcr.io/nvidia/tritonserver:<tag>    # Put the appropriate tag here
  
```

## Install Jupyter notebook inside the Triton container
```
pip3 install jupyter notebook
```

## Starting Jupyter notebook
Assuming the notebook is in `/data` folder,

Start Jupyter notebook by running `jupyter notebook --allow-root --no-browser --port 7001`

