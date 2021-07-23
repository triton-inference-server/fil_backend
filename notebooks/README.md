# Trtion FIL backend with XGBoost

This notebook is a reference for deploying a forest model trained using XGBoost library in Triton Inference Server with Forest Inference Library (FIL) backend. The notebook explains how one can deploy XGBoost model in Triton, check deployment status and send inference requests, set concurrent model execution and dynamic batching and find the best deployment configuration using Model Analyzer.

## Requirements
* Nvidia GPU (Pascal+ Recommended GPUs: T4, V100 or A100)
* [Latest NVIDIA driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* [Docker](https://docs.docker.com/get-docker/)
* [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Run the Triton Inference Server container 

To run this notebook, pull the Triton container or build the container locally as mentioned in the README of [Triton Inference Server FIL backend](https://github.com/triton-inference-server/fil_backend#pre-built-container) Github repo. Before running the container, clone the repository and then run the container:

```
git clone https://github.com/triton-inference-server/fil_backend.git
cd fil_backend/notebooks

docker run \
  -it \
  --gpus=all \
  --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -p 7001:7001 \
  -v $PWD:/data \ 
  nvcr.io/nvidia/tritonserver:<tag>    # Put the appropriate tag here.  
```

## Install Jupyter notebook inside the Triton container
```
pip3 install jupyter
```

## Starting Jupyter notebook
Change directory to `/data` folder and run the jupyter notebook:
```
cd /data
jupyter notebook --allow-root --no-browser --port 7001
```
