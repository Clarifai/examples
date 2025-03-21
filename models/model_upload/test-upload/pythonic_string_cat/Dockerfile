FROM --platform=$TARGETPLATFORM public.ecr.aws/clarifai-models/python-base:3.11 as build

ENV DEBIAN_FRONTEND=noninteractive

#############################
# User specific requirements
#############################
COPY requirements.txt .

# Install requirements and clarifai package and cleanup before leaving this line.
# Note(zeiler): this could be in a future template as {{model_python_deps}}
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir clarifai

# These will be set by the templaing system.
ENV CLARIFAI_PAT=${CLARIFAI_PAT}
ENV CLARIFAI_USER_ID=${CLARIFAI_USER_ID}
ENV CLARIFAI_RUNNER_ID=${CLARIFAI_RUNNER_ID}
ENV CLARIFAI_NODEPOOL_ID=${CLARIFAI_NODEPOOL_ID}
ENV CLARIFAI_COMPUTE_CLUSTER_ID=${CLARIFAI_COMPUTE_CLUSTER_ID}
ENV CLARIFAI_API_BASE=${CLARIFAI_API_BASE}

# Set the NUMBA cache dir to /tmp
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV HOME=/tmp

# Set the working directory to /app
WORKDIR /app

# Copy the current folder into /app/model_dir that the SDK will expect.
# Note(zeiler): would be nice to exclude checkpoints in case they were pre-downloaded.
ARG INCLUDE_MODEL_DIR=true
ARG name=main
RUN if [ "$INCLUDE_MODEL_DIR" = "true" ]; then cp -r . /app/model_dir/main; fi

# Add the model directory to the python path.
ENV PYTHONPATH=${PYTHONPATH}:/app/model_dir/main

ENTRYPOINT ["python", "-m", "clarifai.runners.server"]

# Finally run the clarifai entrypoint to start the runner loop and local dev server.
# Note(zeiler): we may want to make this a clarifai CLI call.
CMD ["--model_path", "/app/model_dir/main"]
