## Text-to-Text Model Examples

These can be used on the fly with minimal or no changes to test deploy all models that take a text input and yield a text output prediction e.g. text generation, summarization and translation models to the Clarifai platform. See the required files section for each model below.

### [hf-model](./hf-model/)

The `hf-model` folder contains code to deploy any `text-generation` of hf.

For instance, suppose you want to deploy [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

Fist of all, download/Clone the checkpoints and store it under the **hf-model/checkpoint** directory.

```bash
$ pip install huggingface-hub
$ huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir hf-model/checkpoint --local-dir-use-symlinksFalse --exclude {EXCLUDED FILE TYPES}
```

Install dependencies to test locally:

```bash
$ pip install -r hf-model/requirements.txt
```

>Note: Package versions may vary for certain models.


### [vLLM](./vllm-model/)

The `vllm-model` folder contains code to deploy `text-generation` using high performance framework `vLLM`.

#### Prerequisites

For instance, suppose you want to deploy [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) using vllm

Fist of all, download/Clone the checkpoints and store it under the **vllm-model/weights/** directory.

```bash
$ pip install huggingface-hub
$ huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir vllm-model/weights/ --local-dir-use-symlinks False --exclude {EXCLUDED FILE TYPES}
```

Then in `inference.py`, you may want to update some config of vllm parameters in LLM(). It is recommended to use `gpu_memory_utilization=0.7`.

## Finally Deploy the model to Clarifai
	
>Note: you can skip testing by setting `--no-test` flag for `build` and `upload` command

1. Build

```bash
$ clarifai build model <path/to/folder>
```

upload `*.clarifai` file to storage to obtain direct download url

2. Upload

```bash
$ clarifai upload model <path/to/folder> --url <your_url> 
```
