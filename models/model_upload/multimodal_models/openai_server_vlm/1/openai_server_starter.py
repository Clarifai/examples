import subprocess
import sys
import threading
from typing import List

from clarifai.utils.logging import logger

PYTHON_EXEC = sys.executable

class OpenAI_APIServer:
  
  def __init__(self, **kwargs):
    self.server_started_event = threading.Event()
    self.process = None
  
  def __del__ (self, *exc):
    # This is important
    # close the server when exit the program
    self.close()

  def close (self):
    if self.process:
      self.process.kill()
  
  def wait_for_startup(self):
    self.server_started_event.wait()
  
  
  def _start_server(self, cmds):  
    try:
      self.process = subprocess.Popen(
          cmds,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      for line in self.process.stderr:
        logger.info(line.strip())
        if f" running on http://{self.host}:" in line.strip():
          self.server_started_event.set()
          break
    except Exception as e:
      if self.process:
        self.process.terminate()
      raise RuntimeError(f"Failed to start Server server: {e}")
  
  
  def start_server_thread(self, cmds:str):
    try:
      # Start the  server in a separate thread
      server_thread = threading.Thread(
          target=self._start_server, args=(cmds,))
      server_thread.start()

      # Wait for the server to start
      self.wait_for_startup()
    except Exception as e:
      raise Exception(e)
  
  
  @classmethod
  def from_lmdeploy_backend(
    cls,
    checkpoints:str,
    server_port=23333, 
    backend:str="turbomind",
    cache_max_entry_count=0.5, 
    tensor_parallel_size = 1, 
    max_prefill_token_num=4096, 
    dtype='float16',
    quantization_format: str = None,
    quant_policy: int = 0,
    chat_template: str = None,
    max_batch_size=16,
    server_name="localhost",
    device="cuda",
    additional_list_args: List[str] = []
  ):
    """Run lmdeploy OpenAI compatible server

    Args:
        checkpoints (str): model id or path
        server_port (int, optional): port. Defaults to 23333.
        backend (str, optional): turbomind or pytorch. Defaults to "turbomind".
        cache_max_entry_count (float, optional): reserved mem for cache. Defaults to 0.5.
        tensor_parallel_size (int, optional): n gpus. Defaults to 1.
        max_prefill_token_num (int, optional): prefill token, the higher the more GPU mems are used. Defaults to 4096.
        dtype (str, optional): dtype. Defaults to 'float16'.
        quantization_format (str, optional): quantization {awq, gptq}. Defaults to None.
        quant_policy (int, optional): KV cache quant policty {0, 4, 8} bits, 0 means not using quantization. Defaults to 0.
        chat_template (str, optional): Chat template. To see all chatempltes, run `lmdeploy list`. Defaults to None.
        max_batch_size (int, optional): batch size. Defaults to 16.
        server_name (str, optional): host name. Defaults to "localhost".
        device (str, optional): device. Defaults to "cuda".
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. Defaults to [].

    """
    # lmdeploy serve api_server $MODEL_DIR --backend $LMDEPLOY_BE --server-port 23333
    cmds = [
      PYTHON_EXEC,
      '-m',
      'lmdeploy',
      'serve',
      'api_server',
      checkpoints,
      '--dtype',
      str(dtype),
      '--backend',
      str(backend),
      '--tp',
      str(tensor_parallel_size),
      '--server-port',
      str(server_port),
      '--server-name',
      str(server_name),
      '--cache-max-entry-count',
      str(cache_max_entry_count),
      '--quant-policy',
      str(quant_policy),
      '--device',
      str(device),
    ]
    
    if quantization_format:
      cmds += ['--model-format', str(quantization_format)]
    
    if chat_template:
      cmds += [ '--chat-template', str(chat_template)]
    
    if max_batch_size:
      cmds += [ '--max-batch-size', str(max_batch_size)]
    
    if max_prefill_token_num:
      cmds += [ '--max-prefill-token-num', str(max_prefill_token_num)]
    
    cmds += additional_list_args
    
    _self = cls()
    
    _self.host = server_name
    _self.port = server_port
    _self.start_server_thread(cmds)
    
    return _self
    
  @classmethod
  def from_vllm_backend(
    cls, 
    checkpoints,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization:float=0.8,
    port=23333,
    host="localhost",
    quantization:str=None,
    additional_list_args: List[str] = []
  ):
    """Run VLLM OpenAI compatible server

    Args:
        checkpoints (str): model id or path
        dtype (str, optional): dtype. Defaults to "float16".
        tensor_parallel_size (int, optional): n gpus. Defaults to 1.
        gpu_memory_utilization (float, optional): % using GPU mem. Defaults to 0.8.
        port (int, optional): port. Defaults to 23333.
        host (str, optional): host name. Defaults to "localhost".
        quantization (str, optional): quantization format {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}. Defaults to None.
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. Defaults to [].

    """
    cmds = [
      PYTHON_EXEC,
      '-m',
      'vllm.entrypoints.openai.api_server',
      '--model',
      checkpoints,
      '--dtype',
      str(dtype),
      '--tensor-parallel-size',
      str(tensor_parallel_size),
      '--gpu-memory-utilization',
      str(gpu_memory_utilization),
      '--port',
      str(port),
      '--host',
      host,
      "--trust-remote-code"
    ]
    if quantization:
      cmds += ['--quantization','awq',]

    cmds += additional_list_args
    
    _self = cls()
    
    _self.host = host
    _self.port = port
    _self.start_server_thread(cmds)
    
    return _self
  
  @classmethod
  def from_sglang_backend(
    cls, 
    checkpoints,
    dtype="float16",
    tp_size=1,
    mem_fraction_static:float=0.8,
    port=23333,
    host="localhost",
    quantization:str=None,
    chat_template: str = None,
    additional_list_args: List[str] = [],
  ):
    
    cmds += additional_list_args
    cmds = [
      PYTHON_EXEC,
      '-m',
      'sglang.launch_server',
      '--model-path',
      checkpoints,
      '--dtype',
      str(dtype),
      '--tp-size',
      str(tp_size),
      '--quantization',
      str(quantization),
      '--mem-fraction-static',
      str(mem_fraction_static),
      '--port',
      str(port),
      '--host',
      host,
      "--trust-remote-code"
    ]
    if chat_template:
      cmds += [
        "--chat-template=", chat_template
      ]
    
    _self = cls()
    
    _self.host = host
    _self.port = port
    _self.start_server_thread(cmds)