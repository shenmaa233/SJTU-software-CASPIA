import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import uuid


class LogManager:
    """Manage loggers for multiple tasks, each with independent log file."""

    def __init__(self, log_dir: str = "./logs", max_tail_bytes: int = 80_000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_tail_bytes = max_tail_bytes
        self._loggers = {}

    def new_session(self, prefix: str = "task-") -> str:
        """Create a new session with its own log file."""
        sid = (prefix + uuid.uuid4().hex)[:12]
        log_file = self.log_dir / f"{sid}.log"

        logger = logging.getLogger(sid)
        logger.setLevel(logging.INFO)

        handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)

        # Avoid duplicate handlers
        if not logger.handlers:
            logger.addHandler(handler)

        self._loggers[sid] = (logger, log_file)
        return sid

    def get_logger(self, sid: str) -> logging.Logger:
        return self._loggers[sid][0]

    def get_log_path(self, sid: str) -> Path:
        return self._loggers[sid][1]

    def read_tail(self, sid: str) -> str:
        """Read the tail of a log file (default 80KB)."""
        log_path = self.get_log_path(sid)
        if not log_path.exists():
            return ""
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > self.max_tail_bytes:
                f.seek(size - self.max_tail_bytes)
            data = f.read()
        return data.decode("utf-8", errors="ignore")
