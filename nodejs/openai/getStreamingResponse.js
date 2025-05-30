import OpenAI from "openai";

// Ensure you have the CLARIFAI_PAT environment variable set with your Clarifai Personal Access Token
const client = new OpenAI({
  baseURL: "https://api.clarifai.com/v2/ext/openai/v1",
  apiKey: process.env.CLARIFAI_PAT,
});

// Stream response from the model
const streamingResponse = await client.chat.completions.create({
  model: "https://clarifai.com/openai/chat-completion/models/gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is photosynthesis?" },
  ],
  stream: true,
});

for await (const part of streamingResponse) {
  if (part.choices?.[0]?.delta.content) {
    console.log(part.choices[0].delta.content);
  }
}
