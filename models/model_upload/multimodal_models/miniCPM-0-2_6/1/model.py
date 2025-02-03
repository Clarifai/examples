import os
from typing import Iterator

import torch
import io
import tempfile
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format
from transformers import (AutoModel, AutoTokenizer, TextIteratorStreamer)
from PIL import Image
from decord import VideoReader, cpu # Decord is a high-performance, video processing library 
import numpy as np
import librosa # Librosa is a python package for music and audio analysis
import soundfile as sf
from moviepy.editor import VideoFileClip # MoviePy is a Python library for video editing
import math


MAX_NUM_FRAMES = 64  # if CUDA OOM occurs, set a smaller number.
AUDIO_SR = 16000      # default sampling rate for audio

def convert_audio_to_bytes(audio_wav, sampling_rate):
    if audio_wav is not None:
        # Create an in-memory buffer
        with io.BytesIO() as buf:
            # Write the audio data to the buffer in WAV format
            sf.write(buf, audio_wav, samplerate=sampling_rate, format="wav")
            # Retrieve the raw bytes from the buffer
            audio_bytes = buf.getvalue()
        return audio_bytes
    return None


def encode_video(video_bytes: bytes):
    """
    Decodes video from bytes and uniformly samples frames up to MAX_NUM_FRAMES.
    Returns a list of PIL.Image frames.
    """
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file.flush()
        video_path = tmp_file.name

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    logger.info(f"Extracted {len(frames)} frames from video.")
    return frames

def get_video_chunk_content(video_bytes, flatten=True):

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file.flush()
        video_path = tmp_file.name

    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)
    audio_np = None
    if video.audio:
      with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
          temp_audio_file_path = temp_audio_file.name
          video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
          audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        if audio_np is not None:
            audio = audio_np[sr*i:sr*(i+1)]
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
        else:
            if flatten:
                contents.extend(["<unit>", image])
            else:
                contents.append(["<unit>", image])
    
    return contents

def decode_audio_from_bytes(audio_bytes: bytes, sr: int = AUDIO_SR,
                            is_pcm: bool = False,
                            pcm_channels: int = 1,
                            pcm_bit_depth: int = 16,
                            ) -> np.ndarray:
    """
    Given bytes for an audio file, decode it into a float32 numpy array at sampling rate sr.
    Supports multiple formats (MP3, WAV, etc.) and does not require ffmpeg.
    """
    if not is_pcm:
        # Common "magic byte" signatures:
        #  - WAV : "RIFF"
        #  - MP3 : Starts with 0xFF 0xFB or "ID3"
        #  - FLAC: "fLaC"
        #  - OGG : "OggS"
        header = audio_bytes[:4]

        if header[:4] == b"RIFF":
            extension = ".wav"
        elif audio_bytes[:2] == b"\xff\xfb" or audio_bytes[:3] == b"ID3":
            extension = ".mp3"
        elif header == b"fLaC":
            extension = ".flac"
        elif header == b"OggS":
            extension = ".ogg"
        # M4A/MP4: can contain "ftyp" near the start
        # For example, "ftypisom", "ftypM4A ", etc.
        # Typically at offset 4, but this can vary. We'll do a simple check:
        elif len(audio_bytes) >= 12 and audio_bytes[4:8] == b'ftyp':
            extension= ".m4a"
        elif audio_bytes[0:4] == b"ftyp" and b"M4A" in audio_bytes[4:12]:
            extension= ".m4a"
        elif audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
            # 0xFF plus one of 0xF2, 0xF3, 0xFA, 0xFB in second byte
            extension= ".mp3"
        else:
            is_pcm = True
          
        if not is_pcm:
          # Write bytes to temporary file
          with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
              tmp_file.write(audio_bytes)
              tmp_file.flush()
              audio_path = tmp_file.name

          # Let librosa load (and internally decode) that file
          audio_np, curr_sr = librosa.load(audio_path, sr=None, mono=True)

          # If you want a specific output sampling rate, resample:
          if sr is not None and curr_sr != sr:
              audio_np = librosa.resample(audio_np, orig_sr=curr_sr, target_sr=sr)

          return audio_np.astype(np.float32)
        else:
          # Fall back to raw PCM decoding
          return decode_raw_pcm(audio_bytes, channels=pcm_channels, bit_depth=pcm_bit_depth)
    else:
        # Decode raw PCM data
        return decode_raw_pcm(audio_bytes, channels=pcm_channels, bit_depth=pcm_bit_depth)

def decode_raw_pcm(
      audio_bytes: bytes,
      channels: int,
      bit_depth: int | str,
    ) -> np.ndarray:

    """
    Decodes raw PCM data from a bytes object into float32 numpy samples in range -1..1.
    Parameters:
    - bit_depth can be 8, 16, 24, 32, or 'float32'.
    If e.g. 16, data is assumed to be signed int16.
    Returns: a 1D numpy float32 array in mono (mix down if channels > 1).
    """
    if bit_depth == 'float32':
      # Raw 32-bit float
      dtype = np.float32
      max_val = 1.0  # Already float
    elif bit_depth == 8:
      # 8-bit unsigned PCM is typical in .wav
      dtype = np.uint8
      max_val = 128.0
    elif bit_depth == 16:
      dtype = np.int16
      max_val = 32768.0
    elif bit_depth == 32:
      # 32-bit integer PCM
      dtype = np.int32
      max_val = 2**31
    else:
      raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")


    arr = np.frombuffer(audio_bytes, dtype=dtype)
    if channels > 1:
        # Reshape to separate channels
        arr = arr.reshape(-1, channels)
        # mixdown to mono by averaging across channels
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if bit_depth == 8:
        arr = (arr - 128.0) / 128.0
    else:
        arr /= max_val
    return arr

# Helper function to convert image bytes to PIL Image
def image_bytes_to_pil(image_bytes):
    """
    Converts image bytes to a PIL Image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Ensure image data is read and decoded
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Unable to convert bytes to image: {e}")

# Helper function to create an output
def create_output(text=None, audio_bytes=None, code=status_code_pb2.SUCCESS):
  """
  Create a single Clarifai Output object containing text and optionally base64 audio data.
  """
  data = resources_pb2.Data()
  if text:
    data.text.raw = text
  if audio_bytes:
    data.audio.base64 = audio_bytes
  output = resources_pb2.Output(data=data, status=status_pb2.Status(code=code))
  return output



def construct_message(input_data: resources_pb2.Data) -> list[dict]:
  """
  Constructs a single multi-modal user message and any relevant advanced chat params
  based on the input data (text, images, audio, video).
  Returns (messages, paramsForChat).
  """
  default_prompt = "Describe the following modalities."
  prompts = []
  images_b = []
  videos_b = []
  audios_b = []

  # If input_data.parts exist, parse them one by one. Each part can be text, image, video, or audio.
  if input_data.parts:
    for part in input_data.parts:
      if part.data.text.raw:
          prompts.append(part.data.text.raw)
      if part.data.image.base64:
          images_b.append(part.data.image.base64)
      if part.data.video.base64:
          videos_b.append(part.data.video.base64)
      if part.data.audio.base64:
          audios_b.append(part.data.audio.base64)
  else:
    # fallback to top-level input_data
    if input_data.text.raw:
        prompts.append(input_data.text.raw)
    if input_data.image.base64:
        images_b.append(input_data.image.base64)
    if input_data.video.base64:
        videos_b.append(input_data.video.base64)
    if input_data.audio.base64:
        audios_b.append(input_data.audio.base64)

  if not prompts:
    prompts.append(default_prompt)

  if len(videos_b) > 1:
    raise ValueError("Only one video is supported at a time.")
  if len(audios_b) > 1:
    raise ValueError("Only one audio is supported at a time.")

  content_list = []

  # 1) Convert images
  for img_bytes in images_b:
      try:
          pil_img = image_bytes_to_pil(img_bytes)
          content_list.append(pil_img)
      except Exception as e:
          logger.error(f"Error decoding image: {e}")

  # 2) Convert video frames
  if videos_b:
      frames = encode_video(videos_b[0])
      content_list.extend(frames)

  # 3) Convert audio
  if audios_b:
      audio_np = decode_audio_from_bytes(audios_b[0], sr=AUDIO_SR)
      content_list.append(audio_np)

  # 4) Add prompts/text
  content_list.extend(prompts)

  # Additional chat params for multi-modal
  chat_params = {}
  # If we have images/video, often we set:
  chat_params["use_image_id"] = False
  # If we're combining many images/frames, we may reduce max_slice_nums if GPU is limited
  chat_params["max_slice_nums"] = 1 if videos_b else 2

  # If multiple modalities (images_b or videos_b or audios_b) are used, you can set "omni_input=True" for the "omni" style
  if images_b or videos_b or audios_b:
      chat_params["omni_input"] = True

  message = {
      "role": "user",
      "content": content_list
  }
  return [message], chat_params


def construct_message_video_streaming(input_data: resources_pb2.Data) -> list[dict]:
    """
    Constructs a single multi-modal user message and any relevant advanced chat params
    based on the input data (text, images, audio, video).
    Returns (messages, paramsForChat).
    """
    prompts = []
    videos_b = []
    audio_b = []
    images_b = []

    messages = []

    # If input_data.parts exist, parse them one by one. Each part can be text, image, video, or audio.
    if input_data.parts:
        for part in input_data.parts:
            if part.data.text.raw:
                prompts.append(part.data.text.raw)
            if part.data.video.base64:
                videos_b.append(part.data.video.base64)
            if part.data.audio.base64:
                audio_b.append(part.data.audio.base64)
            if part.data.image.base64:
                images_b.append(part.data.image.base64)
    else:
        # fallback to top-level input_data
        if input_data.text.raw:
            prompts.append(input_data.text.raw)
        if input_data.video.base64:
            videos_b.append(input_data.video.base64)
        if input_data.audio.base64:
            audio_b.append(input_data.audio.base64)
        if input_data.image.base64:
            images_b.append(input_data.image.base64)

    if len(videos_b) > 1:
        raise ValueError("Only one video is supported at a time.")
    if len(audio_b) > 1:
        raise ValueError("Only one audio is supported at a time.")

    # 1) Add prompts/text
    if prompts:
        messages.append({
            "role": "user",
            "content": prompts
        })

    # 2) Convert video frames
    if videos_b:
        try:
            contents = get_video_chunk_content(videos_b[0], flatten=False)
            messages.extend([{"role": "user", "content": content} for content in contents])
        except Exception as e:
            frames = encode_video(videos_b[0])
            messages.append({
                "role": "user",
                "content": frames
            })
            
            
    if images_b:
        for img_bytes in images_b:
            try:
                pil_img = image_bytes_to_pil(img_bytes)
                messages.append({
                    "role": "user",
                    "content": [pil_img]
                })
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
    if audio_b:
        audio_np = decode_audio_from_bytes(audio_b[0], sr=AUDIO_SR)
        messages.append({
            "role": "user",
            "content": [audio_np]
        })
        
    # Additional chat params for multi-modal
    chat_params = {}
    # If we have images/video, often we set:
    chat_params["use_image_id"] = False
    # If we're combining many images/frames, we may reduce max_slice_nums if GPU is limited
    chat_params["max_slice_nums"] = 1 if videos_b else 2

    # If multiple modalities (images_b or videos_b or audios_b) are used, you can set "omni_input=True" for the "omni" style
    if videos_b:
        chat_params["omni_input"] = True

    return messages, chat_params

# Helper function to get the inference params
def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params


# Helper function to parse the inference params
def parse_inference_params(request):
  default_params = {
  "temperature": 0.7,
  "max_tokens": 512,
  "top_k": 50,
  "top_p": 1.0,
  "do_sample": True,
  # For TTS or not
  "generate_audio": False,
  # For advanced modes
  "return_dict": False,
  "use_tts_template": False
  }
  inference_params = get_inference_params(request)
  parsed = {
      "temperature": inference_params.get("temperature", default_params["temperature"]),
      "max_new_tokens": int(inference_params.get("max_tokens", default_params["max_tokens"])),
      "top_k": int(inference_params.get("top_k", default_params["top_k"])),
      "top_p": inference_params.get("top_p", default_params["top_p"]),
      "do_sample": inference_params.get("do_sample", default_params["do_sample"]),
      "generate_audio": inference_params.get("generate_audio", default_params["generate_audio"]),
      "return_dict": inference_params.get("return_dict", default_params["return_dict"]),
      "session_id": inference_params.get("session_id", ""),
  }
  # If user wants TTS generation, automatically set use_tts_template=True
  if parsed["generate_audio"]:
      parsed["use_tts_template"] = True

  return parsed

def _format_chat_output(chat_output, generate_audio=False):
    """
    Converts the raw chat output into Clarifai-friendly Outputs (text + optional audio).
    """
    out_list = []
    if generate_audio:
        # TTS case
        if isinstance(chat_output, dict):
            text_out = chat_output.get("text", "")
            audio_wav = chat_output.get("audio_wav", None)
            sampling_rate = chat_output.get("sampling_rate", AUDIO_SR)
            if audio_wav is not None:
                audio_bytes = convert_audio_to_bytes(audio_wav, sampling_rate)
                out_list.append(create_output(text=text_out, audio_bytes=audio_bytes))
            else:
                out_list.append(create_output(text=text_out))
        else:
            # Fallback
            out_list.append(create_output(str(chat_output)))
    else:
        # Non-TTS case
        if isinstance(chat_output, dict):
            txt = chat_output.get("text", "")
            out_list.append(create_output(txt))
        elif isinstance(chat_output, list):
            # Possibly multiple strings or dict
            for item in chat_output:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                    out_list.append(create_output(txt))
                else:
                    out_list.append(create_output(str(item)))
        else:
            # Just a single text string
            out_list.append(create_output(str(chat_output)))

    return out_list

class MyRunner(ModelRunner):
  """A custom runner that loads the model and generates text using batched inference."""

  def load_model(self):
    """Load the model here."""
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {self.device}")

    # Load model and tokenizer
    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints, trust_remote_code=True)
    # self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token
    self.model = AutoModel.from_pretrained(
        checkpoints,
        trust_remote_code=True,
        attn_implementation='sdpa', # Use sdpa for faster inference
        low_cpu_mem_usage=True,
        device_map=self.device,
        torch_dtype=torch.bfloat16,
    )
    # Initialize TTS modules
    self.model.init_tts()

    logger.info("Model and tokenizer loaded successfully.")


  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This method generates outputs text for the given inputs using the model."""

    inference_params = parse_inference_params(request)
    all_outputs = []
    generate_audio = inference_params["generate_audio"]

    for input_ in request.inputs:
      input_data = input_.data
      message, extra_chat_params = construct_message(input_data)


      chat_output = self.model.chat(
        msgs=message,
        tokenizer=self.tokenizer,
        sampling=inference_params["do_sample"],
        temperature=inference_params["temperature"],
        max_new_tokens=inference_params["max_new_tokens"],
        top_p=inference_params["top_p"],
        top_k=inference_params["top_k"],
        generate_audio=inference_params["generate_audio"],
        return_dict=inference_params["return_dict"],
        use_tts_template=inference_params.get("use_tts_template", False),
        **extra_chat_params
        )
      outputs = _format_chat_output(chat_output, generate_audio)
      all_outputs.extend(outputs)

    return service_pb2.MultiOutputResponse(
        outputs=all_outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method generates stream of outputs for the given batch of inputs using the model."""

    # Each new stream invocation can be considered a new conversation or session
    session_id = "clarifai_mini_cpm_session"
    self.model.reset_session()  # reset the KV cache for a new 

    inference_params = parse_inference_params(request)
    if len(request.inputs) > 1:
      raise ValueError("Batch generation is not supported in stream mode for this model.")
    # We'll parse the single input as a chunk

    # Each new stream invocation can be considered a new conversation or session
    session_id = "clarifai_mini_cpm_session"
    self.model.reset_session()  # reset the KV cache for a new 

    # Default inference params
    streaming_generate_audio = False
    streaming_use_tts_template = False

    inference_params = parse_inference_params(request)
    streaming_generate_audio = inference_params.get("generate_audio", streaming_generate_audio)
    streaming_use_tts_template = inference_params.get("use_tts_template", streaming_use_tts_template)
    temperature = inference_params["temperature"]
    max_new_tokens = inference_params["max_new_tokens"]
    top_p = inference_params["top_p"]
    top_k = inference_params["top_k"]
    do_sample = inference_params["do_sample"]

    audios = []
    combine_text = ""

    messages, chunk_chat_params = construct_message_video_streaming(request.inputs[0].data)

    for message in messages:
      _ = self.model.streaming_prefill(
          session_id=session_id,
          msgs=[message],
          tokenizer=self.tokenizer,
          **chunk_chat_params
      )
    # Now we call streaming_generate to get partial tokens or partial audio.
    # This is the point where the model actually streams out an answer.
    result_generator = self.model.streaming_generate(
        session_id=session_id,
        tokenizer=self.tokenizer,
        sampling=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        generate_audio=streaming_generate_audio,
        use_tts_template=streaming_use_tts_template,
    )

    if not result_generator:
        # If your model does not truly stream, just return everything at once
        yield service_pb2.MultiOutputResponse(
            outputs=[create_output("Model does not support streaming", code=status_code_pb2.FAILURE)],
            status=status_pb2.Status(code=status_code_pb2.FAILURE)
        )
        return

    if streaming_generate_audio:
        for r in result_generator:
            # r is typically a partial result object with:
            # r.text, r.audio_wav, r.sampling_rate
            partial_text = r.text

            audio_wav = r.audio_wav
            sampling_rate = r.sampling_rate

            audios.append(audio_wav)
            combine_text += partial_text
            if partial_text:
                yield service_pb2.MultiOutputResponse(
                    outputs=[create_output(text=partial_text,)],
                    status=status_pb2.Status(code=status_code_pb2.SUCCESS)
                )
        if audios:
            # Combine all audio chunks into a single audio
            combined_audio = np.concatenate(audios)
            audio_bytes = convert_audio_to_bytes(combined_audio, sampling_rate)
            yield service_pb2.MultiOutputResponse(
                outputs=[create_output(audio_bytes=audio_bytes)],
                status=status_pb2.Status(code=status_code_pb2.SUCCESS)
            )
    else:
        # Text-only streaming
        for r in result_generator:
            partial_text = r["text"]
            yield service_pb2.MultiOutputResponse(
                outputs=[create_output(text=partial_text)],
                status=status_pb2.Status(code=status_code_pb2.SUCCESS)
            )


  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method generates stream of outputs for the given inputs using the model."""
    """
    This method demonstrates how to perform “multimodal live streaming” in a chunked manner
    similar to the official MiniCPM-o documentation.

    Example usage: repeatedly send PostModelOutputsRequest objects with partial frames
    or audio chunks in real time. Then, once you want the partial generation,
    you can pass an inference_param like "action=generate" or similar. Adjust to your
    desired streaming logic.
    """

    # Each new stream invocation can be considered a new conversation or session
    session_id = "clarifai_mini_cpm_session"
    self.model.reset_session()  # reset the KV cache for a new 

    # Default inference params
    streaming_generate_audio = False
    streaming_use_tts_template = False
    temperature = 0.7
    max_new_tokens = 1024
    top_p = 1.0
    top_k = 50
    do_sample = True

    audios = []
    combine_text = ""

    for request in request_iterator:
        # Parse inference_params from each chunk as well

        inference_params = parse_inference_params(request)
        streaming_generate_audio = inference_params.get("generate_audio", streaming_generate_audio)
        streaming_use_tts_template = inference_params.get("use_tts_template", streaming_use_tts_template)
        temperature = inference_params["temperature"]
        max_new_tokens = inference_params["max_new_tokens"]
        top_p = inference_params["top_p"]
        top_k = inference_params["top_k"]
        do_sample = inference_params["do_sample"]

        audios = []
        combine_text = ""

        messages, chunk_chat_params = construct_message_video_streaming(request.inputs[0].data)

        for message in messages:
          _ = self.model.streaming_prefill(
              session_id=session_id,
              msgs=[message],
              tokenizer=self.tokenizer,
              **chunk_chat_params
          )
        # Now we call streaming_generate to get partial tokens or partial audio.
        # This is the point where the model actually streams out an answer.
        result_generator = self.model.streaming_generate(
            session_id=session_id,
            tokenizer=self.tokenizer,
            sampling=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            generate_audio=streaming_generate_audio,
            use_tts_template=streaming_use_tts_template,
        )

        if not result_generator:
            # If your model does not truly stream, just return everything at once
            yield service_pb2.MultiOutputResponse(
                outputs=[create_output("Model does not support streaming", code=status_code_pb2.FAILURE)],
                status=status_pb2.Status(code=status_code_pb2.FAILURE)
            )
            return

        if streaming_generate_audio:
            for r in result_generator:
                # r is typically a partial result object with:
                # r.text, r.audio_wav, r.sampling_rate
                partial_text = r.text

                audio_wav = r.audio_wav
                sampling_rate = r.sampling_rate

                audios.append(audio_wav)
                combine_text += partial_text

                yield service_pb2.MultiOutputResponse(
                    outputs=[create_output(text=partial_text)],
                    status=status_pb2.Status(code=status_code_pb2.SUCCESS)
                )
            if audios:
                # Combine all audio chunks into a single audio
                combined_audio = np.concatenate(audios)
                audio_bytes = convert_audio_to_bytes(combined_audio, sampling_rate)
                yield service_pb2.MultiOutputResponse(
                    outputs=[create_output(audio_bytes=audio_bytes)],
                    status=status_pb2.Status(code=status_code_pb2.SUCCESS)
                )
        else:
            # Text-only streaming
            for r in result_generator:
                partial_text = r["text"]
                yield service_pb2.MultiOutputResponse(
                    outputs=[create_output(text=partial_text)],
                    status=status_pb2.Status(code=status_code_pb2.SUCCESS)
                )


