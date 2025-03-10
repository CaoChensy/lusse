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

> Examples

```python
from lusse.models.manager import VLLMEngineArgs, ServerManager

engine_args = VLLMEngineArgs(
    model="/home/models/Qwen/Qwen2.5-VL-7B-Instruct/",
    port=8111,
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.3,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "介绍图片"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://pic3.zhimg.com/v2-6ac0e399774bde7efb391f76b7f262ca_1440w.jpg",
                },
            },
        ],
    }
],

with ServerManager(engine_args=engine_args, auto_terminate="delay", verbose=True) as serve:
    client = serve.make_client()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        max_tokens=300,
    )

print(response.choices[0])
```

```text
Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='这张图片展示了一本名为《工厂》的书籍的封面。封面上有一幅详细的插图，描绘了一个石砌的工业建筑，可能是18世纪工业革命时期常见的工厂建筑。画面中还有几个人物，似乎在从事一些与纺织或其他工业生产相关的工作。封面的标题是“水力驱动了工业革命？”作者是大卫·麦考利，译者是刘勇军。这本书属于“画给孩子的历史奇迹”系列，通过图像和故事的形式向孩子们介绍历史上的重要事件和时刻。封面上方还标注了书的出版信息，指出是由江苏凤凰少年儿童出版社出版的。', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)
```

----