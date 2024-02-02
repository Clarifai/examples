## Text Embedder Model Examples

These can be used on the fly with minimal or no changes to test deploy text embedding models to the Clarifai platform. See the required files section for each model below.

* ### [Instructor-xl](https://huggingface.co/hkunlp/instructor-xl)

	Requirements to run tests locally:

	* Download/Clone the [huggingface model](https://huggingface.co/hkunlp/instructor-xl) into the **instructor-xl/checkpoint** directory.
	```
	huggingface-cli download hkunlp/instructor-xl --local-dir instructor-xl/checkpoint/sentence_transformers/hkunlp_instructor-xl --local-dir-use-symlinks False
	```

	Deploy the model to Clarifai:
	
	>Note: set `--no-test` flag for `build` and `upload` command to disable testing

	1. Build

	```bash
	$ clarifai build model ./instructor-xl
	```
	
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./instructor-xl --url <your_url> 
	```

