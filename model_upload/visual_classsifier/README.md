## Image Classification Model Examples

These can be used on the fly with minimal or no changes to test deploy image classification models to the Clarifai platform. See the required files section for each model below.

* ### [VIT Age Classifier](./age_vit/)

	Required files to run tests locally:

	* Download the [model checkpoint from huggingface](https://huggingface.co/nateraw/vit-age-classifier/tree/main) and store it under `age_vit/checkpoint/`
	```
	huggingface-cli download nateraw/vit-age-classifier --local-dir age_vit/checkpoint/ --local-dir-use-symlinks False
	```

	Install dependencies to test locally:

	```bash
	$ pip install -r age_vit/requirements.txt
	```

	Deploy the model to Clarifai:
		
	>Note: you can skip testing by setting `--no-test` flag for `build` and `upload` command

	1. Build

	```bash
	$ clarifai build model ./age_vit
	```
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./age_vit --url <your_url> 
	```
