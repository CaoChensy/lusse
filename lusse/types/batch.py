import inspect
from pydantic import BaseModel
from typing import List, Union, Optional
from vllm import SamplingParams
from lusse.types.response import BaseResponse


__all__ = [
    "BatchCompletionRequest",
    "PromptCompletionResponse",
    "BatchCompletionResponse",
]


class BatchCompletionRequest(BaseModel):
    """"""
    model: str
    prompts: List[str]
    temperature: float = 1.0
    top_p: float = 1.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stop_token_ids: Optional[list[int]] = None
    max_tokens: Optional[int] = 16
    min_tokens: int = 0

    def sampling_params(self) -> SamplingParams:
        """"""
        need_params = [name for name, _ in inspect.signature(SamplingParams).parameters.items()]
        params = {key: val for key, val in self.model_dump().items() if key in need_params}
        return SamplingParams(**params)


class PromptCompletionResponse(BaseModel):
    """"""
    text: str
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class BatchCompletionResponse(BaseResponse):
    """"""
    result: List[PromptCompletionResponse]
