import gc
import ray
import time
import torch
import atexit
import threading
import contextlib
from typing import Any
from threading import Lock
from transformers import AutoTokenizer
from dataclasses import asdict
from typing import Dict, Optional, Literal
from vllm import LLM, EngineArgs
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from lusse.utils import LoggerMixin
from lusse.models.manager.base import BaseManager


__all__ = [
    "ModelManager",
    "register_model_configs",
]


class ModelManager(BaseManager, LoggerMixin):
    _instances: Dict[str, 'ModelManager'] = {}
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, engine_args: EngineArgs, model_name: Optional[str] = None, *args, **kwargs):
        """单例模式"""
        model_name = model_name or cls.split_model_name_from_path(model_path=engine_args.model)
        if model_name not in cls._instances:
            with cls._lock:
                if model_name not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(
            self,
            engine_args: EngineArgs,
            model_name: Optional[str] = None,
            auto_terminate: Literal["never", "delay", "directly"] = "delay",
            resource_timeout: Optional[int] = 60,
            verbose: Optional[bool] = False,
            log_file: Optional[str] = None,
            **kwargs,
    ):
        self.auto_terminate = auto_terminate
        self.resource_timeout: int = resource_timeout
        if self._initialized:
            return

        self.model_name = model_name or self.split_model_name_from_path(model_path=engine_args.model)
        self.log_file = log_file or f"./log/{self.model_name}.log"

        self.engine_args = engine_args
        self.verbose: Optional[bool] = verbose
        self.lock: Lock = Lock()
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[Any] = None
        self.release_timer: Optional[threading.Timer] = None
        self._initialized = True

        self._start_logging()

        # 注册强制清理
        self._register_cleanup()

    def _register_cleanup(self):
        """注册多级清理机制"""
        atexit.register(self.stop)

    def _release_resources(self):
        """释放所有相关资源"""
        with self.lock:
            if not self.llm:
                return

            if self.llm:
                try:
                    destroy_model_parallel()
                    destroy_distributed_environment()
                    del self.llm.llm_engine.model_executor
                    del self.llm
                    del self.tokenizer
                    with contextlib.suppress(AssertionError):
                        torch.distributed.destroy_process_group()
                    gc.collect()
                    torch.cuda.empty_cache()
                    ray.shutdown()
                    self.info(f"[{__class__.__name__}] Released resources for model: {self.model_name}")
                finally:
                    self.llm = None
                    self.tokenizer = None

    def reset_shutdown_timer(self, timeout: Optional[int] = None):
        """重置空闲计时器"""
        if timeout is None:
            timeout = self.resource_timeout
        with self.lock:
            if self.release_timer:
                self.release_timer.cancel()
                self.info(msg=f"[{__class__.__name__}] Cancelled existing timer")
            if self.llm:
                self.release_timer = threading.Timer(timeout, self._release_resources)
                self.release_timer.start()
                self.info(msg=f"[{__class__.__name__}] New timer started with {timeout}s")
            else:
                self.warning(msg=f"[{__class__.__name__}] Trying to reset timer without loaded model")

    def load_model(self) -> bool:
        """获取模型实例（自动处理加载和计时器重置）"""
        with self.lock:
            if not self.llm:
                self.info(f"[{__class__.__name__}] Start loading model: {self.model_name}")
                self.llm = LLM(**asdict(self.engine_args))
                self.tokenizer = AutoTokenizer.from_pretrained(self.engine_args.model)
                self.info(f"[{__class__.__name__}] Loaded model successfully: {self.model_name}")
        self.info(f"[{__class__.__name__}] Exit lock for {self.model_name}")
        return True

    @property
    def remaining_time(self) -> float:
        """获取剩余等待时间（秒）"""
        with self.lock:
            if self.release_timer and self.release_timer.is_alive():
                return self.release_timer.interval - (time.time() - self.release_timer.start_time)
            return 0.0

    def stop(self):
        """强制释放资源"""
        self._release_resources()
        self.info(f"[{__class__.__name__}] Shutdown model: {self.model_name}")
        if self.release_timer:
            self.release_timer.cancel()

    def __enter__(self):
        """"""
        self.load_model()
        if self.auto_terminate == "delay":
            self.reset_shutdown_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """"""
        if self.auto_terminate == "directly":
            self.stop()
        else:
            pass


def register_model_configs(config: Dict) -> Dict[str, ModelManager]:
    """"""
    MODEL_CONFIGS = {model_name: EngineArgs(**_config) for model_name, _config in config.items()}
    model_managers = {name: ModelManager(name, args) for name, args in MODEL_CONFIGS.items()}
    return model_managers
