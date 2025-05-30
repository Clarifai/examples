# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reviser agent for correcting inaccuracies based on verified findings."""
import os
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm

from . import prompt

_END_OF_EDIT_MARK = '---END-OF-EDIT---'

# Your Clarifai PAT will be used in alias of OPENAI_API_KEY
clarifai_pat = os.getenv("CLARIFAI_PAT")
if clarifai_pat is None:
    print("Error: CLARIFAI_PAT environment variable not set.")
    exit()
else:
    os.environ["OPENAI_API_KEY"] = clarifai_pat 

def _remove_end_of_edit_mark(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    del callback_context  # unused
    if not llm_response.content or not llm_response.content.parts:
        return llm_response
    for idx, part in enumerate(llm_response.content.parts):
        if _END_OF_EDIT_MARK in part.text:
            del llm_response.content.parts[idx + 1 :]
            part.text = part.text.split(_END_OF_EDIT_MARK, 1)[0]
    return llm_response

MODEL = LiteLlm(model="openai/gcp/generate/models/gemini-2_5-flash",
                      base_url="https://api.clarifai.com/v2/ext/openai/v1",
                      api_key=clarifai_pat)

reviser_agent = Agent(
    model=MODEL,
    name='reviser_agent',
    instruction=prompt.REVISER_PROMPT,
    after_model_callback=_remove_end_of_edit_mark,
)
