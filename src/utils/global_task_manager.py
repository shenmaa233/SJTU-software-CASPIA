# src/utils/global_task_manager.py

"""
Global task manager singleton for the entire application.
This allows different modules (Agent, GEMFactory, etc.) to share the same task queue.
"""

from src.utils import LogManager, TaskRunner

# Global singleton instances
_log_manager = None
_task_runner = None


def get_task_manager() -> TaskRunner:
    """
    Get the global TaskRunner instance (singleton pattern).
    
    Returns:
        The global TaskRunner instance
    """
    global _log_manager, _task_runner
    
    if _task_runner is None:
        _log_manager = LogManager("./logs")
        _task_runner = TaskRunner(_log_manager)
    
    return _task_runner


def get_log_manager() -> LogManager:
    """
    Get the global LogManager instance.
    
    Returns:
        The global LogManager instance
    """
    global _log_manager
    
    if _log_manager is None:
        _log_manager = LogManager("./logs")
    
    return _log_manager

