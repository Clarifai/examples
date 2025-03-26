import base64
from typing import Dict, Iterator, List

from clarifai.runners.utils.data_types import Audio, Image, Video


def build_messages(prompt: str, image: Image, images: List[Image], audio: Audio,
                   audios: List[Audio], video: Video, videos: List[Video],
                   chat_history: List[Dict]) -> List[Dict]:
  """Construct OpenAI-compatible messages from input components."""
  messages = []
  # Add previous conversation history
  if chat_history:
    messages.extend(chat_history)

  if prompt.strip():
    # Build content array for current message
    content = [{'type': 'text', 'text': prompt}]
  # Add single image if present
  if image:
    content.append(_process_image(image))
  # Add multiple images if present
  if images:
    for img in images:
      content.append(_process_image(img))
  # Add single audio if present
  if audio:
    content.append(_process_audio(audio))
  # Add multiple audios if present
  if audios:
    for audio in audios:
      content.append(_process_audio(audio))
  # Add single video if present
  if video:
    content.append(_process_video(video))
  # Add multiple videos if present
  if videos:
    for video in videos:
      content.append(_process_video(video))

  # Append complete user message
  messages.append({'role': 'user', 'content': content})

  return messages


def _process_image(image: Image) -> Dict:
  """Convert Clarifai Image object to OpenAI image format."""
  if image.bytes:
    b64_img = base64.b64encode(image.bytes).decode('utf-8')
    return {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{b64_img}"}}
  elif image.url:
    return {'type': 'image_url', 'image_url': {'url': image.url}}
  else:
    raise ValueError("Image must contain either bytes or URL")


def _process_audio(audio: Audio) -> Dict:
  if audio.bytes:
    audio = base64.b64encode(audio.base64).decode("utf-8")
    audio = {
        "type": "input_audio",
        "input_audio": {
            "data": audio,
            "format": "wav"
        },
    }
  elif audio.url:
    audio = audio.url
    audio = {
        "type": "audio_url",
        "audio_url": {
            "url": audio
        },
    }
  else:
    raise ValueError("Audio must contain either bytes or URL")

  return audio


def _process_video(video: Video) -> Dict:
  if video.bytes:
    video = "data:video/mp4;base64," + \
        base64.b64encode(video.base64).decode("utf-8")
    video = {
        "type": "video_url",
        "video_url": {
            "url": video
        },
    }
  elif video.url:
    video = video.url
    video = {
        "type": "video_url",
        "video_url": {
            "url": video
        },
    }
  else:
    raise ValueError("Video must contain either bytes or URL")

  return video


class OpenAIWrapper:

  def __init__(self, client: object, modalities: List[str] = None):
    self.client = client
    self.modalities = modalities or []
    self._validate_modalities()
    self.model_id = self._get_model_id()

  def _validate_modalities(self):
    valid_modalities = {'image', 'audio', 'video'}
    invalid = set(self.modalities) - valid_modalities
    if invalid:
      raise ValueError(f"Invalid modalities: {invalid}. Valid options: {valid_modalities}")

  def _get_model_id(self):
    try:
      return self.client.models.list().data[0].id
    except Exception as e:
      raise ConnectionError("Failed to retrieve model ID from API") from e

  @staticmethod
  def make_api_url(host: str, port: int, version: str = "v1") -> str:
    return f"http://{host}:{port}/{version}"

  def predict(self,
              prompt: str = "",
              image: Image = None,
              images: List[Image] = None,
              audio: Audio = None,
              audios: List[Audio] = None,
              video: Video = None,
              videos: List[Video] = None,
              chat_history: List[Dict] = None,
              max_tokens: int = 512,
              temperature: float = 0.7,
              top_p: float = 0.8) -> str:
    """Process a single request through OpenAI API."""
    messages = build_messages(prompt, image, images or [], audio, audios or [], video, videos or
                              [], chat_history or [])
    response = self.client.chat.completions.create(
        model=self.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False)

    return response.choices[0].message.content

  def generate(self,
               prompt: str,
               image: Image = None,
               images: List[Image] = None,
               audio: Audio = None,
               audios: List[Audio] = None,
               video: Video = None,
               videos: List[Video] = None,
               chat_history: List[Dict] = None,
               max_tokens: int = 512,
               temperature: float = 0.7,
               top_p: float = 0.8) -> Iterator[str]:
    """Stream response from OpenAI API."""
    messages = build_messages(prompt, image, images or [], audio, audios or [], video, videos or
                              [], chat_history or [])

    stream = self.client.chat.completions.create(
        model=self.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True)

    for chunk in stream:
      if chunk.choices:
        text = (chunk.choices[0].delta.content
                if (chunk and chunk.choices[0].delta.content) is not None else '')
        yield text
