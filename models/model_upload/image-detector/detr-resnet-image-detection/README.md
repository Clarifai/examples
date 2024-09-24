

# Example model directory

This is an example model directory for a dockerized model and helps to create image-detector runner

There are only three necessary files:
 - requirements.txt for python requirements
 - Dockerfile - the requirements to include in the docker image.
 - model.py - the implementation of the model.

In future there may be an additional config.yaml added as a required file that collects other
 information together and we may get rid of Dockerfile replacing it with a full templating.


## Build

# Add your HF_TOKEN as environment variable
```bash
export HF_TOKEN=""
```

```bash
docker build --no-cache -t luv-image-detector-runner:latest --build-arg TARGET_PLATFORM="linux/amd64" --build-arg DRIVER_VERSION=530 --build-arg CUDA_VERSION=12.1.0 --build-arg PYTHON_VERSION=3.10 --build-arg HF_TOKEN=$HF_TOKEN --progress=plain  .
```

## Run

To test your build image locally you can do:
```
docker run --gpus=all -it --rm  \
       -e CLARIFAI_PAT=$CLARIFAI_PAT \
       -e CLARIFAI_USER_ID=$CLARIFAI_USER_ID \
       -e CLARIFAI_RUNNER_ID=$CLARIFAI_RUNNER_ID \
       -e CLARIFAI_NODEPOOL_ID=$CLARIFAI_NODEPOOL_ID \
       -e CLARIFAI_COMPUTE_CLUSTER_ID=$CLARIFAI_COMPUTE_CLUSTER_ID \
       -e CLARIFAI_API_BASE=$CLARIFAI_API_BASE \
       luv-image-detector-runner:latest
```
