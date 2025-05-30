# Clarifai Open AI Example

Clarifai provides an OpenAI-compatible API endpoint, which allows you to leverage your existing OpenAI API code and workflows to make inferences with Clarifai models, including those that integrate or wrap OpenAI models.

The built-in compatibility layer converts your OpenAI calls directly into Clarifai API requests, letting you harness Clarifai's diverse models as custom tools in your OpenAI projects.

This simplifies the integration process, as you don't need to rewrite your code specifically for Clarifai's native API structure if you're already familiar with OpenAI's.

## Prerequisites

- A Clarifai account with API access
- A Clarifai Personal Access Token (PAT)

## Setup

Clone this directory to your local machine using [degit](https://github.com/Rich-Harris/degit)

```bash
degit clarifai/examples/nodejs/openai
```

Install the required packages:

```bash
npm install
```

Add your Clarifai PAT to the environment variable `CLARIFAI_PAT`

```bash
export CLARIFAI_PAT=your_pat_here
```

or use a `.env` file to setup your environment variables.

## Notes

- The example uses Clarifai's GPT-4o model by default, but you can easily switch to other models
- Make sure your Clarifai PAT has the necessary permissions
- The script uses async/await for better performance
- Tool responses are processed to provide natural language output

