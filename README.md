# 模型资源管理器

这是一个简单的`vLLM`模型资源管理器，用于管理多个模型实例，并支持动态加载和卸载模型。用途是在多用户场景下，多模型在同一GPU上运行，每个用户可以动态加载自己的模型，即时释放显存资源，而不需要担心资源竞争和内存溢出的问题。

这包括两个管理器：``ModelManager``和``EngineManager``。

**vLLM Model Manager**

> 管理多个模型实例，支持动态加载和卸载模型。

*推荐使用 `with` 进行资源与模型的生命周期管理*

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

# easy mode
with ModelManager(engine_args=args) as model:
    model.load_model()
    model.llm.generate(...)
```

*或者手动进行资源管理*

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

# or manual mode
model_manager = ModelManager(args, verbose=True)
model_manager.load_model()

# 模型推理
model_manager.llm.generate(...)

# 模型资源将在5s后自动释放
model_manager.reset_shutdown_timer(5)

# 手动释放模型资源
model_manager.stop()
```

**vLLM Serve Manager**

> 管理多个模型实例，支持动态加载和卸载以VLLM方式启动的OpenAI服务。

*推荐使用 `with` 进行资源与模型的生命周期管理*

```python
from lusse.models.manager import VLLMEngineArgs, ServerManager

engine_args = VLLMEngineArgs(
    model="/home/models/Qwen/Qwen2.5-0.5B-Instruct/",
    port=8102,
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.2,
    enforce_eager=True,
    enable_auto_tool_choice=True,
    tool_call_parser="hermes",
)

# easy mode
with ServerManager(engine_args=engine_args) as serve:
    client = serve.make_client()
    messages = [{"role": "user", "content": "你好"}]
    response = client.chat.completions.create(model="Qwen/Qwen2.5-0.5B-Instruct", messages=messages)
    print(serve.server_name)
    print(response.choices[0].message.content)
```

*或者手动进行资源管理*


```python
from lusse.models.manager import VLLMEngineArgs, ServerManager

engine_args = VLLMEngineArgs(
    model="/home/models/Qwen/Qwen2.5-0.5B-Instruct/",
    port=8102,
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.2,
    enforce_eager=True,
    enable_auto_tool_choice=True,
    tool_call_parser="hermes",
)

# or manual mode
serve = ServerManager(engine_args=engine_args, verbose=True)

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

-----