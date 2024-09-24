# Example model directory

This is an exmaple of model directory for a dockerized Openai Whisper model.
 
There are only three necessary files:
 - requirements.txt for python requirements
 - model.py - the implementation of the model.
 - config.yaml - contains compute resource information and other details needed to build the Docker image and upload the model to the to dev/ prod.

## Upload Model to dev
Refer to the [Compute Orchestration: Uploading your first model to dev](https://clarifai.atlassian.net/wiki/spaces/EN/pages/3693150243/Compute+Orchestration+Uploading+your+first+model+to+dev) doc for guidance on uploading and deploying your model to the dev environment.

## Run Model locally

### Get Dockerfile
First, obtain the Dockerfile in the example folder by running `get_dockerfile.py` with the model directory path.

```bash
python get_dockerfile.py --folder whisper-speech-recognition
```

### Build the docker image

```bash
docker build -t runner-whisper-speech-recognition:latest .
```

### Run docker image 

To test your build image locally you can do:
```
docker run -it --rm  \
       -e CLARIFAI_PAT=$CLARIFAI_PAT \
       -e CLARIFAI_USER_ID=$CLARIFAI_USER_ID \
       -e CLARIFAI_RUNNER_ID=$CLARIFAI_RUNNER_ID \
       -e CLARIFAI_NODEPOOL_ID=$CLARIFAI_NODEPOOL_ID \
       -e CLARIFAI_COMPUTE_CLUSTER_ID=$CLARIFAI_COMPUTE_CLUSTER_ID \
       -e CLARIFAI_API_BASE=$CLARIFAI_API_BASE \
       runner-whisper-speech-recognition:latest
```
