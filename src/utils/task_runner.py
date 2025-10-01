import threading
import subprocess
from typing import Callable, Dict, Any
from src.utils import LogManager


def run_command(cmd: str, logger) -> None:
    """
    Run a shell command and stream its stdout/stderr to logger in real time.
    """
    process = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in process.stdout:
        logger.info(line.strip())
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}: {cmd}")


class TaskRunner:
    """Manage background tasks with independent logging."""

    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def start(self, task_fn: Callable, *args, prefix: str = "task-") -> str:
        """
        Start a task in background.
        - task_fn must accept (logger, *args).
        - Returns a unique session id (sid).
        """
        sid = self.log_manager.new_session(prefix=prefix)
        logger = self.log_manager.get_logger(sid)

        self.tasks[sid] = {"done": False, "success": None, "result": None}

        def wrapper():
            try:
                result = task_fn(logger, *args)
                self.tasks[sid]["done"] = True
                self.tasks[sid]["success"] = True
                self.tasks[sid]["result"] = result
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                self.tasks[sid]["done"] = True
                self.tasks[sid]["success"] = False
                self.tasks[sid]["result"] = None

        t = threading.Thread(target=wrapper, daemon=True)
        t.start()
        return sid

    def poll(self, sid: str):
        """
        Poll task state.
        Returns (logs, status, result)
        """
        if sid not in self.tasks:
            return "", "⚠️ Unknown task", ""
        meta = self.tasks[sid]
        logs = self.log_manager.read_tail(sid)
        if meta["done"]:
            if meta["success"]:
                return logs, "✅ Done", meta["result"] or ""
            else:
                return logs, "❌ Failed", ""
        else:
            return logs, "🚧 Running...", ""
