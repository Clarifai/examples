## Text to Image Model Examples

These can be used on the fly with minimal or no changes to test deploy text to image models to the Clarifai platform. See the required files section for each model below.

* ### [sd-v1.5 (Stable-Diffusion-v1.5)](./sd-v1.5/)

	* Download the [model checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and store it under `sd-v1.5/checkpoint`
	
	```bash
	$ pip install huggingface-hub
	$ huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir sd-v1.5/checkpoint --local-dir-use-symlinks False --exclude *.safetensors *.non_ema.bin *.fp16.bin *.ckpt
	```

	Deploy the model to Clarifai:
	
	>Note: set `--no-test` flag for `build` and `upload` command to disable testing

	1. Build

	```bash
	$ clarifai build model ./sd-v1.5
	```
	
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./sd-v1.5 --url <your_url> 
	```
