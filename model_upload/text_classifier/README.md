## Text Classification Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy text classification models to the Clarifai platform. See the required files section for each model below.

* ### [XLM-Roberta Tweet Sentiment Classifier](./xlm-roberta/)

	Required files to run tests locally:

	* Download the [model checkpoint](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment/tree/main) and store it under `xlm-roberta/checkpoint/`

	```
	$ pip install huggingface-hub
	$ huggingface-cli download cardiffnlp/twitter-xlm-roberta-base-sentiment --local-dir xlm-roberta/checkpoint/ --local-dir-use-symlinks False --exclude *.msgpack *.h5
	```

	Deploy the model to Clarifai:
	
	>Note: set `--no-test` flag for `build` and `upload` command to disable testing

	1. Build

	```bash
	$ clarifai build model ./xlm-roberta
	```
	
	upload `*.clarifai` file to storage to obtain direct download url

	2. Upload

	```bash
	$ clarifai upload model ./xlm-roberta --url <your_url> 
	```

