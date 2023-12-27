
# Langchain Agents

This Notebook examples will showcases the different scenario and use cases where you can use Langchain's [Agents](https://python.langchain.com/docs/modules/agents/) framework
to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code).
In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.

## Contents
### 1. Retrieval QA with Conversation memory

The notebook walks you through the high level concept and idea to build a chatbot Doc Q/A in using clarifai as vectorstore and langchain's conversationbuffermemory. This enables the user to have a context of recent chat history in buffer as well as helping the user to query about document stored in the DB.

### 2. Doc-retrieve using Langchain ReAct Agent

The notebook gives a walkthrough to build a Doc Q/A using clarifai vectorstore and langchain's React Docstore with Webscraped docs of Clarifai. This enables the user to retrive info regarding the Docs of Clarifai.

The steps are as follows:

- Websracping from Clarifai Docs Website.
- Processing and Storing the Docs in Clarifai Vectorstore.
- Building a React Agent to search in the Clarifai vectorstore.
- Using the Agent to answer for User queries related to Clarifai.




















## Documentation
Check out the docs in the below reference links to deep dive into clarifai and langchain.

**Langchain**: [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)

**Clarifai**: [https://www.clarifai.com](https://www.clarifai.com/)

**Clarifai Demo**: [https://clarifai.com/demo](https://clarifai.com/demo)

**Sign up for a free Account**: [https://clarifai.com/signup](https://clarifai.com/signup)

**Developer Guide**: [https://docs.clarifai.com](https://docs.clarifai.com/)

**Clarifai Community**: [https://clarifai.com/explore](https://clarifai.com/explore)

**Python SDK Docs**: [https://docs.clarifai.com/python-sdk/api-reference](https://docs.clarifai.com/python-sdk/api-reference)
