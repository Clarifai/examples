## Visual Embedding Model Examples

These can be used on the fly with minimal or no changes to test deploy visual embedding models to the Clarifai platform. See the required files section for each model below.

* ### [vit-base](./vit-base/)

	Requirements to run tests locally:

	Download the [model checkpoint & sentencepiece bpe model from huggingface](https://huggingface.co/google/vit-base-patch16-224/tree/main) and store it under `vit-base/checkpoint`
	```
	huggingface-cli download google/vit-base-patch16-224 --local-dir vit-base/checkpoint --local-dir-use-symlinks False --exclude *.msgpack *.h5 *.safetensors
	```
	
	Install dependecies to test locally

	```bash
	$ pip install -r vit-base/requirements.txt
	```
	
	Deploy the model to Clarifai:
	
	>Note: set `--no-test` flag for `build` and `upload` command to disable testing

	1. Build

	```bash
	$ clarifai build model ./vit-base
	```
	
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./vit-base --url <your_url> 
	```