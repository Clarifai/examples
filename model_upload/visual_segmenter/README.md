## Visual Segmentation Model Examples

These can be used on the fly with minimal or no changes to test deploy visual segmentation models to the Clarifai platform. See the required files section for each model below.

* ### [segformer-b2](./segformer-b2/)

	Requirements to run tests locally:

	Download/Clone the [huggingface model](https://huggingface.co/mattmdjaga/segformer_b2_clothes) into the **segformer-b2/checkpoint** directory.
	```
	$ huggingface-cli download mattmdjaga/segformer_b2_clothes --local-dir segformer-b2/checkpoint --local-dir-use-symlinks False --exclude *.safetensors optimizer.pt
	```
	
	Install dependecies to test locally

	```bash
	$ pip install -r segformer-b2/requirements.txt
	```
	
	Deploy the model to Clarifai:
	
	>Note: set `--no-test` flag for `build` and `upload` command to disable testing

	1. Build

	```bash
	$ clarifai build model ./segformer-b2
	```
	
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./segformer-b2 --url <your_url> 
	```
