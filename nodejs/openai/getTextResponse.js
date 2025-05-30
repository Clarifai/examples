import OpenAI from "openai";

// Ensure you have the CLARIFAI_PAT environment variable set with your Clarifai Personal Access Token
const client = new OpenAI({
  baseURL: "https://api.clarifai.com/v2/ext/openai/v1",
  apiKey: process.env.CLARIFAI_PAT,
});

// Example usage of the OpenAI client with Clarifai's GPT-4o model
const response = await client.chat.completions.create({
  model: "https://clarifai.com/openai/chat-completion/models/gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is photosynthesis?" },
  ],
});

console.log(response.choices?.[0]?.message.content);
