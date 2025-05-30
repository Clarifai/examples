![image](https://github.com/user-attachments/assets/b22c9807-f5e7-49eb-b00d-598e400781af)


# Clarifai Python SDK and Integrations Examples


This is a collection of examples for the [clarifai-python](https://github.com/Clarifai/clarifai-python) SDK and Integrations done with Clarifai SDK. Use these examples to learn Clarifai and build your own robust and scalable AI applications.

Experience the power of Clarifai in building Computer Vision , Natual Language processsing , Generative AI applications.

[Website](https://www.clarifai.com/) | [Schedule Demo](https://www.clarifai.com/company/schedule-demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/python-sdk/api-reference) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)


## Setup
* Sign up for a free account at [clarifai](https://clarifai.com/signup) and set your PAT key.
* You can generate PAT key in the Personal settings -> [Security section](https://clarifai.com/settings/security)

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

## Compute Orchestration and Agentic Examples
| Function    | Link    | Description | Open in Colab |
| ----------- | ----------- | -----------   | -----------   |
| Basics      | [Basics](basics/basics.ipynb) | Basic Functionalities (create, list, patch, delete) of SDK App, Dataset, Input & Model Classes | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/basics/basics.ipynb) |
| Compute Orchestration      | [CRUD Operations](ComputeOrchestration/crud_operations.ipynb) | Basic Functionalities (create, list, get, delete) of Compute Orchestration Classes - ComputeCluster, Nodepool, Deployment | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/ComputeOrchestration/crud_operations.ipynb) |
| Model Predict  | [Model Predict](models/model_predict.ipynb) | Prediction Functionalities of SDK Model Class for different type of input data | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_predict.ipynb) |
|                | [Using OpenAI Client | Use the openAI client to call openAI-compatible models in Clarifai |  |
|                | [Using LiteLLM]() | Use Litellm to call openAI-compatible models in Clarifai |  |
|                | [Model Predict Javascript]() | Examples coming soon  |  |
| Model Upload  | [Model Upload]([models/model_predict.ipynb](https://github.com/clarifai/runners-examples)) | Upload custom models, MCP tools and more in our new runner examples repo for compute orchestration! |  |
| MCP Tools      | [MCP Tool Examples](https://github.com/Clarifai/runners-examples/tree/main/mcp) | Upload custom MCP tools and have them fully hosted in Clarifai to use with any MCP client. |  |
| Agent Toolkits | [CrewAI Examples](https://github.com/Clarifai/examples/tree/main/agents/CrewAI) | Build agents with CrewAI toolkits on top of Clarifai LLMs and MCP tools. |  |
|                | [Google ADK](https://github.com/Clarifai/examples/tree/main/agents/Google-ADK)  | Create agetns with Google ADK leveraging LLMs and tools powered by Clarifai | | 
|                | [LLM + MCP Examples](https://github.com/Clarifai/examples/tree/main/agents/mcp) | Simple python native examples of building agents covering function calls, JSON parsing and more |  |


## SDK Notebooks
| Function    | Notebook    | Description | Open in Colab |
| ----------- | ----------- | -----------   | -----------   |
| Basics      | [Basics](basics/basics.ipynb) | Basic Functionalities (create, list, patch, delete) of SDK App, Dataset, Input & Model Classes | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/basics/basics.ipynb) |
| Compute Orchestration      | [CRUD Operations](ComputeOrchestration/crud_operations.ipynb) | Basic Functionalities (create, list, get, delete) of Compute Orchestration Classes - ComputeCluster, Nodepool, Deployment | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/ComputeOrchestration/crud_operations.ipynb) |
| Concepts      | [Concept Management](concepts/concept_management.ipynb) | Basic Functionalities (add, search, delete) of Concept Relations | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/concepts/concept_management.ipynb) |
| Datasets | [Basics](datasets/basics.ipynb) | Basic Functionalities of Dataset & Input Class in SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/datasets/basics.ipynb) |
|             | [Input Upload](datasets/upload/input_upload.ipynb) | Upload Functionalities of SDK with different kinds of data (image, text, audio, video ) and annotations (classes, bbox, etc) using Input Class | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/datasets/upload/input_upload.ipynb) |
|             | [Dataset Upload](datasets/upload/dataset_upload.ipynb) | Upload Functionalities of SDK with different of dataset annotation formats (Clarifai, Cifar10, VOC, etc.) using Dataset Class | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/datasets/upload/dataset_upload.ipynb) |
|             | [Dataset Export](datasets/export/dataset_export.ipynb) | Export Functionalities of SDK to different of dataset annotation formats (Clarifai, Cifar10, VOC, etc.) using Dataset Class | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/datasets/export/dataset_export.ipynb) |
| Model Predict  | [Model Predict](models/model_predict.ipynb) | Prediction Functionalities of SDK Model Class for different type of input data | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_predict.ipynb) |
|   Workflows   | [Create Workflow](workflows/create_workflow.ipynb) | Different kinds of Workflow Creation examples using SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/workflows/create_workflow.ipynb) |
|             | [Patch Workflow](workflows/patch_workflow.ipynb) | Modifying a workflow with patch operations using SDK Workflow Class | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/workflows/patch_workflow.ipynb) |
|             | [Export Workflow](workflows/export_workflow.ipynb) | Exporting Workflow config and create a modified workflow using SDK Workflow Class | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/workflows/export_workflow.ipynb) |
| Model Training  | [Image Classification Training](models/model_train/image-classification_training.ipynb) | Model Train demo for Visual-Classifier model type with MMClassification_EfficientNet Template | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-classification_training.ipynb) |
|             | [Text Classification Training](models/model_train/text-classification_training.ipynb) | Model Train demo for Text-Classifier model type with HF_GPTNeo_125m_lora template | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/text-classification_training.ipynb) |
|             | [Image Detection Training](models/model_train/image-detection_training.ipynb) | Model Train demo for Visual-Detector model type with MMDetection template | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-detection_training.ipynb) |
|             | [Image Segmentation Training](models/model_train/image-segmentation_training.ipynb) | Model Train demo for Visual-Segmenter model type with MMDetection template | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/image-segmentation_training.ipynb) |
|             | [Transfer Learn Training](models/model_train/transfer-learn.ipynb) | Model Train demo for Embedded-Classifier model type using Transfer-Learning | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_train/transfer-learn.ipynb) |
| Model Evaluation  | [Embedding Classifier](models/model_eval/embedding_classifier_eval.ipynb) | Model Eval demo for Embedded-Classifier model type using SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_eval/embedding_classifier_eval.ipynb) |
| | [Text Classifier](models/model_eval/text_classification_eval.ipynb) | Model Eval demo for Text-Classifier model type using SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_eval/text_classification_eval.ipynb) |
| | [Visual Classifier](models/model_eval/visual_classifier_eval.ipynb) | Model Eval demo for Visual-Classifier model type using SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_eval/visual_classifier_eval.ipynb) |
| | [Visual Detector](models/model_eval/visual_detector_eval.ipynb) | Model Eval demo for Visual-Detector model type using SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/models/model_eval/visual_detector_eval.ipynb) |
| Search      | [Vector Search](search/cross_modal_search.ipynb) | Introductory guide to setting up a Cross-Modal Search system using SDK | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/search/cross_modal_search.ipynb) |
| RAG      | [RAG](RAG/RAG.ipynb) | RAG setup and chat with the RAG interface. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/RAG/RAG.ipynb) |



## CLI Notebooks
| Function    | Notebook    | Description | Open in Colab |
| ----------- | ----------- | -----------   | -----------   |
| Compute Orchestration      | [Compute Orchestration](CLI/compute_orchestration.ipynb) | Basic functionalities of Compute Orchestration using CLI | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/CLI/compute_orchestration.ipynb) |
|   Model   | [Model Predict](CLI/model.ipynb) | Model Prediction using CLI | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/CLI/model.ipynb) |



## Integrations Notebooks
| Integration | Function    | Notebook    | Open in Colab |
| ----------- | ----------- | ----------- | -----------   |
| [Langchain](https://python.langchain.com/docs/get_started/introduction)   | Chains      | [Prompt Templates and Chains](Integrations/Langchain/Chains/Prompt-templates_and_chains.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/Prompt-templates_and_chains.ipynb) |
|             |             | [Retrieval QA Chain](Integrations/Langchain/Chains/Retrieval_QA_chain_with_Clarifai_Vectorstore.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/Retrieval_QA_chain_with_Clarifai_Vectorstore.ipynb) |
|             |             | [Router Chain](Integrations/Langchain/Chains/Router_chain_examples_with_Clarifai_SDK.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/Router_chain_examples_with_Clarifai_SDK.ipynb) |
|             |             | [PostgreSQL LLM](Integrations/Langchain/Chains/PostgreSQL_LLM.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Chains/PostgreSQL_LLM.ipynb) |
|             | Agents       | [Conversational Agent](Integrations/Langchain/Agents/Retrieval_QA_with_Conversation_memory.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Agents/Retrieval_QA_with_Conversation_memory.ipynb) |
|             |             | [ReAct Docstore Agent](Integrations/Langchain/Agents/Doc-retrieve_using_Langchain-ReAct_Agent.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Langchain/Agents/Doc-retrieve_using_Langchain-ReAct_Agent.ipynb) |
| [DeepEval](https://github.com/confident-ai/deepeval) |LLM Evaluation| [LLM Evaluation](Integrations/DeepEval/DeepEval_clarifai_evaluation_example.ipynb)|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/DeepEval/DeepEval_clarifai_evaluation_example.ipynb) |
| [Unstructured.io](https://unstructured.io/)   | | [Github Data Ingestion](Integrations/Unstructured/Clarifai_github_using_unstructured_io_integration_example.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Unstructured/Clarifai_github_using_unstructured_io_integration_example.ipynb) |
|             |             | [S3 Data Ingestion](Integrations/Unstructured/Clarifai_Unstructured_integration_demo.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Unstructured/Clarifai_Unstructured_integration_demo.ipynb) |
|             |             | [DropBox Data Ingestion](Integrations/Unstructured/Dropbox_Clarifai_Unstructured_integration_example.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Integrations/Unstructured/Dropbox_Clarifai_Unstructured_integration_example.ipynb) |


## Data Utils Notebooks
| Data Util            | Function            | Notebook    | Open in Colab |
| ------------------| ------------------- | ----------- | -----------   |
| Image   | [Image Annotation ](Data_Utils/Image%20Annotation/)   | [annotation loader](Data_Utils/Image%20Annotation/image_annotation_loader.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Data_Utils/Image%20Annotation/image_annotation_loader.ipynb) |
| Multimodal   | [Data Ingestion ](Data_Utils/Ingestion%20pipelines/)   | [Ready to Use Pipelines](Data_Utils/Ingestion%20pipelines/Ready_to_use_foundational_pipelines.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Data_Utils/Ingestion%20pipelines/Ready_to_use_foundational_pipelines.ipynb) |
|     | [Data Ingestion ](Data_Utils/Ingestion%20pipelines/)   | [Multimodal Ingestion](Data_Utils/Ingestion%20pipelines/Multimodal_dataloader.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Data_Utils/Ingestion%20pipelines/Multimodal_dataloader.ipynb) |
| | [Data Ingestion ](Data_Utils/Ingestion%20pipelines/) |[Advanced Multimodal Ingestion with summarizer](Data_Utils/Ingestion%20pipelines/Multimodal_ingest_RAG_notebook.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clarifai/examples/blob/main/Data_Utils/Ingestion%20pipelines/Multimodal_ingest_RAG_notebook.ipynb) |

## Note

Although these scripts are run on your local machine, they'll communicate with Clarifai and run in our cloud on demand.

Examples provide a guided tour through Clarifai's concepts and capabilities.
contains uncategorized, miscellaneous examples.
