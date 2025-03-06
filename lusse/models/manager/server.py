import os
import time
import signal
import atexit
import threading
import subprocess
from openai import OpenAI
from vllm import EngineArgs
from dataclasses import fields, dataclass
from typing import get_origin, get_args, Dict, Optional, Any, Union, List
from threading import Thread, Timer, Lock
from lusse.utils import LoggerMixin


__all__ = [
    "VLLMEngineArgs",
    "ServerManager",
]


@dataclass
class VLLMEngineArgs(EngineArgs):
    """"""
    port: int = 8000
    enable_auto_tool_choice: bool = False
    tool_call_parser: Optional[str] = None

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
        return cmd


class ServerManager(LoggerMixin):
    _instances: Dict[str, 'ServerManager'] = {}
    _lock: Lock = Lock()

    def __new__(cls, server_name: str, engine_args: VLLMEngineArgs, *args, **kwargs):
        with cls._lock:
            if server_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[server_name] = instance
        return cls._instances[server_name]

    def __init__(
            self,
            server_name: str,
            engine_args: VLLMEngineArgs,
            verbose: Optional[bool] = False,
            log_file: str = "vllm_server.log",
            resource_timeout: int = 60,
            server_startup_timeout: int = 300,
    ):
        if self._initialized:
            return

        self.server_name: str = server_name
        self.engine_args: VLLMEngineArgs = engine_args
        self.verbose: Optional[bool] = verbose
        self.log_file: Optional[str] = log_file
        self.resource_timeout: int = resource_timeout                # 资源回收最大时间（秒）
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
        self.stop()

    def start(self,) -> bool:
        """启动OpenAI兼容服务（阻塞直至启动完成）

        Returns:
            bool: 服务是否启动成功
        """
        with self.lock:
            if self.is_running:
                self.warning(f"[{__class__.__name__}] Service already running")
                return True  # 已运行视为成功

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
            if not ready:
                error_msg = f"[{__class__.__name__}] Service failed to start within {self.server_startup_timeout}s"
                self.error(error_msg)
                self.stop()
                raise TimeoutError(error_msg)

            if self.start_error:
                raise RuntimeError(f"[{__class__.__name__}] Service startup failed: {self.start_error}")

            self.is_running = True
            return True

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
            try:
                # 终止进程组
                if self.process:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.process.wait(timeout=10)
                    self.process = None
                self.is_running = False
                self.info(f"[{__class__.__name__}] Service stopped successfully")
            except Exception as e:
                self.error(f"[{__class__.__name__}] Error stopping service: {str(e)}")
            finally:
                if self.shutdown_timer:
                    self.shutdown_timer.cancel()

    def _cleanup(self) -> None:
        """资源清理"""
        self.stop()
        self.close_handlers()

    # 上下文管理器支持
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
