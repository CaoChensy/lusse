from pydantic import BaseModel, ConfigDict


__all__ = [

]


class VLLMEngineConfig(BaseModel):
    """"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

