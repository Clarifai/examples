import { createOpenAI } from "@ai-sdk/openai";
import { generateText, tool } from "ai";
import { z } from "zod";

const openai = createOpenAI({
  baseURL: "https://api.clarifai.com/v2/ext/openai/v1",
  apiKey: process.env.CLARIFAI_PAT,
});

const model = openai(
  "https://clarifai.com/openai/chat-completion/models/gpt-4o",
);

const result = await generateText({
  model,
  tools: {
    weather: tool({
      description: "Get the weather in a location",
      parameters: z.object({
        location: z.string().describe("The location to get the weather for"),
      }),
      execute: async ({ location }) => ({
        location,
        temperature: 72 + Math.floor(Math.random() * 21) - 10,
      }),
    }),
    cityAttractions: tool({
      parameters: z.object({ city: z.string() }),
    }),
  },
  prompt:
    "What is the weather in San Francisco and what attractions should I visit?",
});

console.log(result.toolResults);
