import threading
import subprocess
from typing import Callable, Dict, Any, List, Tuple
from datetime import datetime
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
    """Manage background tasks with independent logging and metadata tracking."""

    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def start(self, task_fn: Callable, *args, prefix: str = "task-", 
              task_name: str = "Unnamed Task", task_type: str = "general") -> str:
        """
        Start a task in background with metadata.
        
        Args:
            task_fn: Function to execute (must accept logger as first argument)
            *args: Arguments to pass to task_fn
            prefix: Prefix for session ID
            task_name: Human-readable task name
            task_type: Task type category (e.g., "gem_build", "gene_annotation")
        
        Returns:
            Unique session ID (sid)
        """
        sid = self.log_manager.new_session(prefix=prefix)
        logger = self.log_manager.get_logger(sid)

        self.tasks[sid] = {
            "done": False,
            "success": None,
            "result": None,
            "name": task_name,
            "type": task_type,
            "start_time": datetime.now(),
            "end_time": None
        }

        def wrapper():
            try:
                logger.info(f"🚀 Task started: {task_name}")
                result = task_fn(logger, *args)
                self.tasks[sid]["done"] = True
                self.tasks[sid]["success"] = True
                self.tasks[sid]["result"] = result
                self.tasks[sid]["end_time"] = datetime.now()
                logger.info(f"✅ Task completed successfully")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                self.tasks[sid]["done"] = True
                self.tasks[sid]["success"] = False
                self.tasks[sid]["result"] = None
                self.tasks[sid]["end_time"] = datetime.now()

        t = threading.Thread(target=wrapper, daemon=True)
        t.start()
        return sid

    def poll(self, sid: str) -> Tuple[str, str, str]:
        """
        Poll task state.
        
        Returns:
            Tuple of (logs, status, result)
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

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all tasks.
        
        Returns:
            List of task metadata dictionaries with sid included
        """
        tasks_list = []
        for sid, meta in self.tasks.items():
            task_info = {
                "sid": sid,
                "name": meta.get("name", "Unknown"),
                "type": meta.get("type", "general"),
                "status": "✅ Done" if meta["done"] and meta["success"] else 
                         "❌ Failed" if meta["done"] and not meta["success"] else 
                         "🚧 Running...",
                "start_time": meta.get("start_time", "N/A"),
                "end_time": meta.get("end_time", "N/A") if meta["done"] else "N/A",
                "result": meta.get("result", "")
            }
            tasks_list.append(task_info)
        # Sort by start time, newest first
        tasks_list.sort(key=lambda x: x["start_time"] if isinstance(x["start_time"], datetime) else datetime.min, 
                       reverse=True)
        return tasks_list

    def get_task_info(self, sid: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific task.
        
        Args:
            sid: Session ID
            
        Returns:
            Task metadata dictionary
        """
        if sid not in self.tasks:
            return {}
        meta = self.tasks[sid]
        return {
            "sid": sid,
            "name": meta.get("name", "Unknown"),
            "type": meta.get("type", "general"),
            "done": meta["done"],
            "success": meta.get("success"),
            "result": meta.get("result", ""),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time")
        }
