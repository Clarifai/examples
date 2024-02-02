## Multimodal Embedder Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy image classification models to the Clarifai platform. See the required files section for each model below.

* ### [CLIP](./clip/)

	Required files to run tests locally:

	* Download the [model checkpoint from huggingface](https://huggingface.co/openai/clip-vit-base-patch32) and store it under `clip/1/checkpoint/`


	```
	$ pip install huggingface-hub
	$ huggingface-cli download openai/clip-vit-base-patch32 --local-dir clip/checkpoint/ --local-dir-use-symlinks False --exclude *.msgpack *.h5
	```

	Deploy the model to Clarifai:
	
	>Note: set `--no-test` flag for `build` and `upload` command to disable testing

	1. Build

	```bash
	$ clarifai build model ./clip
	```
	
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./clip --url <your_url> 
	```
