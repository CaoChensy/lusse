__all__ = [
    "BaseManager"
]


class BaseManager:
    """"""

    @staticmethod
    def split_model_name_from_path(model_path: str) -> str:
        """"""
        items = [item for item in model_path.split("/") if item != ""]
        return f"{items[-2]}/{items[-1]}"
