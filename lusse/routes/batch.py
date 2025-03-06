from typing import Any
from fastapi import HTTPException
from lusse import app
from lusse.types import BatchCompletionRequest, BatchCompletionResponse, PromptCompletionResponse


__all__ = [
    "batch_completion"
]


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    """"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


@app.post("/batch/completion")
async def batch_completion(
    request: BatchCompletionRequest
):
    """"""
    model_manager = app.state.model_managers.get(request.model)
    try:
        llm, tokenizer = model_manager.load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load Model Error, Error Message:\n{str(e)}")
    try:
        texts = [apply_chat_template(tokenizer, prompt=prompt) for prompt in request.prompts]
        outputs = llm.generate(texts, request.sampling_params())
        model_manager.start_timeout(60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generate Error, Error Message:\n{str(e)}")

    result = [PromptCompletionResponse(
        text=sample.outputs[0].text,
        finish_reason=sample.outputs[0].finish_reason,
        stop_reason=sample.outputs[0].stop_reason,
    ) for sample in outputs]
    return BatchCompletionResponse(result=result)
