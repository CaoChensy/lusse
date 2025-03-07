import os
import time
import signal
import atexit
import threading
import subprocess
from openai import OpenAI
from vllm import EngineArgs
from dataclasses import fields, dataclass
from typing import get_origin, get_args, Dict, Optional, Union, List, Literal
from threading import Thread, Timer, Lock
from lusse.utils import LoggerMixin
from lusse.models.manager.base import BaseManager


__all__ = [
    "VLLMEngineArgs",
    "ServerManager",
]


@dataclass
class VLLMEngineArgs(BaseManager, EngineArgs):
    """"""
    port: int = 8000
    enable_auto_tool_choice: bool = False
    tool_call_parser: Optional[str] = None

    def validate_serve_model_name(self, cmd: list[str]) -> list[str]:
        """"""
        if '--served-model-name' not in cmd:
            cmd.append('--served-model-name')
            cmd.append(self.split_model_name_from_path(model_path=self.model))
        return cmd

    def to_command(self) -> List[str]:
        cmd = ["vllm", "serve", self.model]
        for field in fields(self):
            name = field.name.replace('_', '-')
            value = getattr(self, field.name)
            if field.name == "model":
                continue

            if value is None or value == field.default:
                continue

            # 类型检查（支持Optional[bool]）
            is_optional_bool = (
                    get_origin(field.type) is Union and
                    bool in get_args(field.type) and
                    type(None) in get_args(field.type)
            )
            is_bool = field.type == bool

            if is_bool or is_optional_bool:
                # 处理None和False情况
                if value is None or value is False:
                    continue
                # 只处理True的情况
                cmd.append(f"--{name}")
                continue
            cmd.extend([f"--{name}", str(value)])

        cmd = self.validate_serve_model_name(cmd)
        return cmd


class ServerManager(BaseManager, LoggerMixin):
    _instances: Dict[str, 'ServerManager'] = {}
    _lock: Lock = Lock()

    def __new__(cls, engine_args: VLLMEngineArgs, server_name: Optional[str] = None, *args, **kwargs):
        """"""
        server_name = server_name or cls.split_model_name_from_path(model_path=engine_args.model)
        with cls._lock:
            if server_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[server_name] = instance
        return cls._instances[server_name]

    def __init__(
            self,
            engine_args: VLLMEngineArgs,
            server_name: Optional[str] = None,
            auto_terminate: Literal["never", "delay", "directly"] = "delay",
            verbose: Optional[bool] = False,
            log_file: Optional[str] = None,
            resource_timeout: int = 60,
            server_startup_timeout: int = 300,
            **kwargs,
    ):
        self.auto_terminate = auto_terminate
        self.resource_timeout: int = resource_timeout
        if self._initialized:
            return

        self.server_name = server_name or self.split_model_name_from_path(model_path=engine_args.model)
        self.log_file = log_file or f"./log/{self.server_name}.log"

        self.engine_args: VLLMEngineArgs = engine_args
        self.verbose: Optional[bool] = verbose
        self.server_startup_timeout: int = server_startup_timeout    # 服务器最大启动时间（秒）

        self.lock: Lock = Lock()
        self.process: Optional[subprocess.Popen] = None
        self.health_check_timer: Optional[Timer] = None
        self.shutdown_timer: Optional[Timer] = None
        self.is_running: bool = False
        self._initialized = True

        # 服务启动事件
        self.ready_event = None
        self.start_error = None

        self._start_logging()

        # 注册退出处理
        atexit.register(self._cleanup)

    def _health_check(self) -> bool:
        """检查服务是否正常运行"""
        if self.process is None or self.process.poll() is not None:
            self.start_error = f"[{__class__.__name__}] Service process not running"
            self.warning(msg=self.start_error)
            return False
        try:
            import requests
            response = requests.get(f"http://localhost:{self.engine_args.port}/health")
            return response.status_code == 200
        except Exception:
            return False

    def _monitor_service(self) -> None:
        """服务监控线程"""
        self.info(f"[{__class__.__name__}] Starting health checks...")
        start_time = time.time()

        while time.time() - start_time < self.server_startup_timeout:
            if self.process is None or self.process.poll() is not None:
                self.start_error = f"[{__class__.__name__}] Service process not running, at time: {time.time() - start_time:.3f}s"
                self.warning(msg=self.start_error)
                break
            try:
                if self._health_check():
                    self.info(f"[{__class__.__name__}] Health check passed")
                    self.ready_event.set()  # 通知主进程启动完成
                    return
            except Exception as e:
                self.start_error = str(e)

            time.sleep(1)

        # 超时或失败时触发事件
        self.ready_event.set()
        self.error(f"[{__class__.__name__}] Health check failed. Service not responding.")
        Thread(target=self.stop, daemon=True).start()

    def start(self,) -> bool:
        """启动OpenAI兼容服务（阻塞直至启动完成）

        Returns:
            bool: 服务是否启动成功
        """
        with self.lock:
            if self.is_running:
                self.warning(f"[{__class__.__name__}] Service already running")
                return self.is_running  # 已运行视为成功

            command = self.engine_args.to_command()

            # 创建启动完成事件
            self.ready_event = threading.Event()
            self.start_error = None

            # 启动服务进程
            self.process = subprocess.Popen(
                command,
                stdout=open(self.log_file, 'a'),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # 创建新进程组
            )

            self.info(f"[{__class__.__name__}] Starting service with PID: {self.process.pid}")

            # 启动健康检查：启动监控线程
            monitor_thread = Thread(target=self._monitor_service, daemon=True)
            monitor_thread.start()

            # 等待启动完成或超时
            ready = self.ready_event.wait(timeout=self.server_startup_timeout)
            if not ready or self.start_error:
                error_msg = f"[{__class__.__name__}] Service failed to start within {self.server_startup_timeout}s, or Service startup failed: {self.start_error}"
                self.error(error_msg)
                need_stop = True
            else:
                need_stop = False
                self.is_running = True
        if need_stop:
            self.stop()
        return self.is_running

    def start_with_timeout(self, timeout: Optional[int] = None) -> bool:
        """
        启动OpenAI兼容服务（阻塞直至启动完成或超时）
        :param timeout: 启动后的超时时间回收资源，单位为秒
        :return:
        """
        if timeout is None:
            timeout = self.resource_timeout

        running = self.start()
        self.reset_shutdown_timer(timeout)
        return running

    def reset_shutdown_timer(self, timeout: int) -> None:
        """重置自动关闭计时器"""
        with self.lock:
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
            self.shutdown_timer = Timer(timeout, self.stop)
            self.shutdown_timer.start()
            self.info(f"[{__class__.__name__}] Shutdown timer reset to {timeout}s")

    def make_client(
            self,
            host: Optional[str] = None,
            port: Optional[Union[str, int]] = None,
            api_key: Optional[str] = "EMPTY"
    ) -> OpenAI:
        """发送请求并重置计时器"""
        if not self.is_running:
            raise RuntimeError(f"[{__class__.__name__}] Service not running")

        if host is None:
            host = "localhost"
        if port is None:
            port = self.engine_args.port
        base_url = f"http://{host}:{port}/v1"
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client

    def stop(self) -> None:
        """优雅地停止服务"""
        with self.lock:
            if not self.is_running:
                return
            self.is_running = False
            process = self.process  # 保存进程引用
            self.process = None     # 清空进程引用
            shutdown_timer = self.shutdown_timer
            self.shutdown_timer = None
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                self.info(f"[{__class__.__name__}] Service stopped successfully")
            except Exception as e:
                self.error(f"[{__class__.__name__}] Error stopping service: {str(e)}")
        if shutdown_timer:
            shutdown_timer.cancel()

    def _cleanup(self) -> None:
        """资源清理"""
        self.stop()
        self.close_handlers()

    def __enter__(self):
        """"""
        if self.auto_terminate == "delay":
            self.start_with_timeout()
        else:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """"""
        if self.auto_terminate == "directly":
            self._cleanup()
        else:
            pass
