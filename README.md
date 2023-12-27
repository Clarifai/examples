# Clarifai Python SDK and Integrations Examples


This is a collection of examples for the [clarifai-python](https://github.com/Clarifai/clarifai-python) SDK and Integrations done with Clarifai SDK. Use these examples to learn Clarifai and build your own robust and scalable AI applications.

Experience the power of Clarifai in building Computer Vision , Natual Language processsing , Generative AI applications.

## Setup
* Sign up for a free account at [clarifai](https://clarifai.com/signup) and set your PAT key.

* Install the [Clarifai python sdk.](https://github.com/Clarifai/clarifai-python/tree/master)

* Export your PAT as an environment variable.
    ```cmd
    export CLARIFAI_PAT={your personal access token}
    ```

* Explore and run the examples  in this  repo.

## Usage

The examples are organized into several folders based on their category. A quick example below,

```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="text_clf", split="train", module_dir="path_to_imdb_reviews_module")
```


## SDK Notebooks
| Function    | Notebook    | Open in Colab |
| ----------- | ----------- | -----------   |
| Basics      | [Basics](basics/basics.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/basics/basics.ipynb) |
| Data Upload | [Input Upload](datasets/upload/input_upload.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/datasets/upload/input_upload.ipynb) |
|             | [Dataset Upload](datasets/upload/dataset_upload.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/datasets/upload/dataset_upload.ipynb) |
|   Workflows   | [Create Workflow](workflows/create_workflow.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/workflows/create_workflow.ipynb) |
|             | [Export Workflow](workflows/export_workflow.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/workflows/export_workflow.ipynb) |
| Model Predict  | [Model Predict](models/model_predict.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_predict.ipynb) |
| Model Training  | [Image Classification Training](models/model_train/image-classification_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-classification_training.ipynb) |
|             | [Text Classification Training](models/model_train/text-classification_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/text-classification_training.ipynb) |
|             | [Image Detection Training](models/model_train/image-detection_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-detection_training.ipynb) |
|             | [Image Segmentation Training](models/model_train/image-segmentation_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-segmentation_training.ipynb) |
|             | [Transfer Learn Training](models/model_train/transfer-learn.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/transfer-learn.ipynb) |
| Search      | [Vector Search](search/cross_modal_search.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/search/cross_modal_search.ipynb) |







## Integrations Notebooks
| Integration | Function    | Notebook    | Open in Colab |
| ----------- | ----------- | ----------- | -----------   |
| [Langchain](https://python.langchain.com/docs/get_started/introduction)   | Chains      | [Prompt Templates and Chains](Integrations/Langchain/Chains/Prompt-templates_and_chains.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/Prompt-templates_and_chains.ipynb) |
|             |             | [Retrieval QA Chain](Integrations/Langchain/Chains/Retrieval_QA_chain_with_Clarifai_Vectorstore.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/Retrieval_QA_chain_with_Clarifai_Vectorstore.ipynb) |
|             |             | [Router Chain](Integrations/Langchain/Chains/Router_chain_examples_with_Clarifai_SDK.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/Router_chain_examples_with_Clarifai_SDK.ipynb) |
|             | Agents       | [Conversational Agent](Integrations/Langchain/Agents/Retrieval_QA_with_Conversation_memory.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Agents/Retrieval_QA_with_Conversation_memory.ipynb) |
|             |             | [ReAct Docstore Agent](Integrations/Langchain/Agents/Doc-retrieve_using_Langchain-ReAct_Agent.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Agents/Doc-retrieve_using_Langchain-ReAct_Agent.ipynb) |


## Note

Although these scripts are run on your local machine, they'll communicate with Clarifai and run in our cloud on demand.

Examples provide a guided tour through Clarifai's concepts and capabilities.
contains uncategorized, miscellaneous examples.
