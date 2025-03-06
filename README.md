# 模型资源管理器

> vLLM Model Manager

```python
from vllm import EngineArgs
from lusse.models.manager import ModelManager

args = EngineArgs(
        model="/home/models/Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.05,
        max_num_seqs=2,
        max_model_len=1024,
        max_num_batched_tokens=8192,
        tensor_parallel_size=1,
        enforce_eager=True,
)
name = "Qwen/Qwen2.5-0.5B-Instruct"

model_manager = ModelManager(name, args, verbose=True)
model_manager.load_model()

# 模型推理
model_manager.llm.generate(...)

# 模型资源将在5s后自动释放
model_manager.reset_shutdown_timer(5)

# 手动释放模型资源
model_manager.stop()
```

> vLLM Serve Manager

```python
from lusse.models.manager import VLLMEngineArgs, ServerManager

engine_args = VLLMEngineArgs(
    model="/home/models/Qwen/Qwen2.5-0.5B-Instruct/",
    port=8102,
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.2,
    served_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    enforce_eager=True,
    enable_auto_tool_choice=True,
    tool_call_parser="hermes",
)

serve = ServerManager(
    "qwen_14b_server", 
    engine_args=engine_args, 
    log_file="./Qwen2.5-0.5B-Instruct.log", 
    verbose=True)

# 启动服务
running = serve.start()

# openai client
client = serve.make_client()

messages = [{"role": "user", "content": "你好"}]
response = client.chat.completions.create(model="Qwen/Qwen2.5-0.5B-Instruct", messages=messages)
print(client)

# 模型资源将在5s后自动释放
serve.reset_shutdown_timer(5)

# 停止服务
serve.stop()
```
