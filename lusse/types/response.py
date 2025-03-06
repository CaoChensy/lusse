from pydantic import BaseModel, Field
from typing import Optional, Any, Literal


__all__ = [
    "BaseResponse",
]


class BaseResponse(BaseModel):
    """"""
    code: Literal[200, 400, 500] = Field(
        default=200,
        description="""
                200 OK - 请求成功。
                400 Bad Request - 请求无效，服务器无法理解。
                500 Internal Server Error - 服务器内部错误。
            """)
    message: Literal["success", "fail"] = Field(
        default="success",
        description="操作结果信息。success表示操作成功，fail表示操作失败。")
    result: Optional[Any] = Field(
        default=None, description="返回结果信息")
