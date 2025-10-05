# Task and Logging System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

---

## Overview

The Task and Logging System provides a unified infrastructure for managing long-running background tasks with independent logging capabilities. This system is designed to handle asynchronous operations across different modules (Agent, GEMFactory, etc.) while maintaining organized, searchable logs for each task.

### Key Features

- ✅ **Asynchronous Task Execution**: Run long-running tasks in background threads
- 📝 **Independent Logging**: Each task gets its own log file with automatic rotation
- 🔄 **Real-time Monitoring**: Poll task status and read logs while tasks are running
- 🌐 **Global Singleton Pattern**: Share task manager across entire application
- 📊 **Task Metadata Tracking**: Track task name, type, status, start/end times
- 🔍 **Log Tailing**: Efficiently read recent logs without loading entire files

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Global Task Manager                        │
│              (Singleton Pattern - Shared)                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   LogManager     │◄────────┤   TaskRunner     │          │
│  │                  │         │                  │          │
│  │  - Creates logs  │         │  - Manages tasks │          │
│  │  - Rotates files │         │  - Threading     │          │
│  │  - Reads tails   │         │  - Status track  │          │
│  └──────────────────┘         └──────────────────┘          │
│           │                            │                     │
└───────────┼────────────────────────────┼─────────────────────┘
            │                            │
            ▼                            ▼
    ┌──────────────┐          ┌──────────────────┐
    │  Log Files   │          │  Background       │
    │  ./logs/     │          │  Threads          │
    │  task-*.log  │          │  (Daemon)         │
    └──────────────┘          └──────────────────┘
```

---

## Components

### 1. LogManager (`src/utils/log_manager.py`)

Manages individual log files for tasks with automatic rotation.

**Key Responsibilities:**
- Create unique session IDs for each task
- Set up rotating file handlers
- Provide logger instances
- Read log file tails efficiently

### 2. TaskRunner (`src/utils/task_runner.py`)

Executes tasks in background threads with status tracking.

**Key Responsibilities:**
- Start tasks asynchronously
- Track task metadata (name, type, status, times)
- Poll task status and logs
- Manage task lifecycle

### 3. Global Task Manager (`src/utils/global_task_manager.py`)

Provides singleton instances accessible throughout the application.

**Key Responsibilities:**
- Initialize and maintain single LogManager instance
- Initialize and maintain single TaskRunner instance
- Provide easy access via `get_task_manager()` and `get_log_manager()`

---

## Quick Start

### Basic Task Execution

```python
from src.utils import get_task_manager

# Get the global task manager
task_manager = get_task_manager()

# Define your task function (logger is always first parameter)
def my_task(logger, param1, param2):
    logger.info(f"Starting task with {param1} and {param2}")
    # Do some work...
    result = param1 + param2
    logger.info(f"Task completed with result: {result}")
    return result

# Start the task
task_id = task_manager.start(
    my_task,
    "value1", "value2",
    prefix="mytask-",
    task_name="My Custom Task",
    task_type="computation"
)

print(f"Task started with ID: {task_id}")
```

### Checking Task Status

```python
# Poll task status
logs, status, result = task_manager.poll(task_id)

print(f"Status: {status}")
print(f"Result: {result}")
print(f"Recent logs:\n{logs}")
```

---

## Detailed Usage

### 1. Using LogManager Directly

If you need fine-grained control over logging:

```python
from src.utils import get_log_manager

# Get the global log manager
log_manager = get_log_manager()

# Create a new logging session
session_id = log_manager.new_session(prefix="myapp-")

# Get the logger for this session
logger = log_manager.get_logger(session_id)

# Use the logger
logger.info("Application started")
logger.warning("This is a warning")
logger.error("An error occurred")

# Read recent logs
recent_logs = log_manager.read_tail(session_id)
print(recent_logs)

# Get log file path
log_path = log_manager.get_log_path(session_id)
print(f"Full logs at: {log_path}")
```

### 2. Running Shell Commands with Logging

Use the `run_command` helper for executing shell commands:

```python
from src.utils import get_task_manager, run_command

def my_shell_task(logger, command):
    """Task that runs a shell command with logging."""
    logger.info(f"Executing command: {command}")
    try:
        run_command(command, logger)
        logger.info("Command completed successfully")
        return "Success"
    except RuntimeError as e:
        logger.error(f"Command failed: {e}")
        raise

# Start the task
task_id = task_manager.start(
    my_shell_task,
    "ls -la /home",
    prefix="shell-",
    task_name="List Directory",
    task_type="shell"
)
```

### 3. Task Metadata and Monitoring

Get information about all tasks or specific tasks:

```python
from src.utils import get_task_manager

task_manager = get_task_manager()

# Get all tasks
all_tasks = task_manager.get_all_tasks()
for task in all_tasks:
    print(f"ID: {task['sid']}")
    print(f"Name: {task['name']}")
    print(f"Type: {task['type']}")
    print(f"Status: {task['status']}")
    print(f"Start: {task['start_time']}")
    print(f"End: {task['end_time']}")
    print(f"Result: {task['result']}")
    print("-" * 50)

# Get specific task info
task_info = task_manager.get_task_info(task_id)
if task_info:
    print(f"Task {task_id} is {'done' if task_info['done'] else 'running'}")
    if task_info['done']:
        print(f"Success: {task_info['success']}")
        print(f"Result: {task_info['result']}")
```

---

## Integration Examples

### Example 1: Bioinformatics Pipeline Task

```python
from src.utils import get_task_manager
import subprocess

def gene_annotation_task(logger, genome_file, output_dir):
    """
    Run gene annotation using GeneMarkS.
    
    Args:
        logger: Logger instance (automatically provided)
        genome_file: Path to genome FASTA file
        output_dir: Output directory path
    
    Returns:
        Path to annotation results
    """
    logger.info(f"Starting gene annotation for {genome_file}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Validate input
        logger.info("Validating input file...")
        if not os.path.exists(genome_file):
            raise FileNotFoundError(f"Genome file not found: {genome_file}")
        
        # Step 2: Run annotation
        logger.info("Running GeneMarkS...")
        cmd = f"gms2.pl --seq {genome_file} --output {output_dir}/output.gff"
        
        process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream output to logger
        for line in process.stdout:
            logger.info(f"[stdout] {line.strip()}")
        
        for line in process.stderr:
            logger.warning(f"[stderr] {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"GeneMarkS failed with code {process.returncode}")
        
        # Step 3: Verify output
        logger.info("Verifying output files...")
        output_file = f"{output_dir}/output.gff"
        if os.path.exists(output_file):
            logger.info(f"✅ Annotation completed: {output_file}")
            return output_file
        else:
            raise RuntimeError("Output file not generated")
            
    except Exception as e:
        logger.error(f"❌ Task failed: {str(e)}")
        raise

# Usage
task_manager = get_task_manager()

task_id = task_manager.start(
    gene_annotation_task,
    "/data/genome.fna",
    "/data/output",
    prefix="gene-",
    task_name="E. coli Genome Annotation",
    task_type="gene_annotation"
)

print(f"Gene annotation task started: {task_id}")
```

### Example 2: Data Processing with Progress Logging

```python
from src.utils import get_task_manager
from tqdm import tqdm

def process_dataset(logger, input_file, output_file):
    """
    Process a large dataset with progress logging.
    
    Args:
        logger: Logger instance
        input_file: Input CSV file
        output_file: Output CSV file
    
    Returns:
        Statistics dictionary
    """
    import pandas as pd
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows")
    
    results = []
    errors = 0
    
    logger.info("Processing data...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Process each row
            processed = process_row(row)
            results.append(processed)
            
            # Log progress every 1000 rows
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} rows")
        
        except Exception as e:
            errors += 1
            logger.warning(f"Error processing row {idx}: {e}")
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    stats = {
        "total_rows": len(df),
        "successful": len(results),
        "errors": errors
    }
    
    logger.info(f"✅ Processing complete: {stats}")
    return stats

# Usage
task_manager = get_task_manager()

task_id = task_manager.start(
    process_dataset,
    "input_data.csv",
    "output_data.csv",
    prefix="proc-",
    task_name="Dataset Processing",
    task_type="data_processing"
)
```

### Example 3: Integration with LangChain Agent Tools

```python
from langchain_core.tools import tool
from src.utils import get_task_manager
from typing import Dict

# Get global task manager
_task_manager = get_task_manager()

@tool
def submit_analysis_task(data_file: str, analysis_type: str) -> Dict[str, str]:
    """
    Submit a data analysis task.
    
    Args:
        data_file: Path to data file
        analysis_type: Type of analysis to perform
    
    Returns:
        Dict with task_id and status message
    """
    def analysis_task(logger, file_path, analysis):
        logger.info(f"Starting {analysis} analysis on {file_path}")
        # Perform analysis...
        result = perform_analysis(file_path, analysis, logger)
        logger.info(f"Analysis complete: {result}")
        return result
    
    task_id = _task_manager.start(
        analysis_task,
        data_file,
        analysis_type,
        prefix="analysis-",
        task_name=f"{analysis_type} Analysis",
        task_type="analysis"
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": f"Analysis task submitted. Task ID: {task_id}"
    }

@tool
def check_analysis_status(task_id: str) -> Dict[str, str]:
    """Check the status of an analysis task."""
    logs, status, result = _task_manager.poll(task_id)
    
    return {
        "task_id": task_id,
        "status": status,
        "result": result,
        "recent_logs": logs[-1000:] if logs else ""
    }
```

### Example 4: Web UI Integration (Gradio)

```python
import gradio as gr
from src.utils import get_task_manager

def submit_job(file_input, options):
    """Submit a job from web UI."""
    task_manager = get_task_manager()
    
    task_id = task_manager.start(
        my_processing_function,
        file_input.name,
        options,
        prefix="web-",
        task_name=f"Web Job: {file_input.name}",
        task_type="web_submission"
    )
    
    return f"Job submitted! Task ID: {task_id}"

def refresh_status(task_id_input):
    """Refresh task status in web UI."""
    task_manager = get_task_manager()
    logs, status, result = task_manager.poll(task_id_input)
    
    return status, logs, result

def list_all_jobs():
    """List all jobs in web UI."""
    task_manager = get_task_manager()
    tasks = task_manager.get_all_tasks()
    
    # Format for display
    task_list = []
    for task in tasks:
        task_list.append([
            task['sid'],
            task['name'],
            task['status'],
            str(task['start_time'])
        ])
    
    return task_list

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Tab("Submit Job"):
        file_input = gr.File(label="Upload File")
        options_input = gr.Textbox(label="Options")
        submit_btn = gr.Button("Submit")
        output = gr.Textbox(label="Result")
        
        submit_btn.click(submit_job, [file_input, options_input], output)
    
    with gr.Tab("Monitor Tasks"):
        refresh_btn = gr.Button("Refresh")
        task_table = gr.Dataframe(
            headers=["Task ID", "Name", "Status", "Start Time"],
            label="All Tasks"
        )
        
        refresh_btn.click(list_all_jobs, None, task_table)

demo.launch()
```

---

## Best Practices

### 1. Logger Usage

✅ **DO:**
- Always accept `logger` as the first parameter in task functions
- Use appropriate log levels: `info`, `warning`, `error`
- Log important milestones and progress updates
- Include relevant context in log messages

❌ **DON'T:**
- Don't use `print()` statements - use `logger.info()` instead
- Don't log sensitive information (passwords, API keys)
- Don't log too verbosely in tight loops (use sampling)

```python
# ✅ Good
def good_task(logger, file_path):
    logger.info(f"Processing file: {file_path}")
    for i, item in enumerate(items):
        if i % 100 == 0:  # Log every 100 items
            logger.info(f"Progress: {i}/{len(items)}")
    logger.info("Processing complete")

# ❌ Bad
def bad_task(file_path):  # Missing logger parameter!
    print("Processing...")  # Using print instead of logger
    for item in items:
        print(f"Item: {item}")  # Too verbose
```

### 2. Task Function Design

✅ **DO:**
- Make task functions focused and single-purpose
- Return meaningful results
- Handle exceptions gracefully
- Validate inputs early

```python
def well_designed_task(logger, input_file, output_dir):
    """
    Well-designed task function.
    
    Args:
        logger: Logger instance
        input_file: Path to input file
        output_dir: Output directory
    
    Returns:
        Dict with output file paths and statistics
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If output_dir is invalid
    """
    # Validate inputs
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input not found: {input_file}")
    
    if not os.path.isdir(output_dir):
        raise ValueError(f"Invalid output directory: {output_dir}")
    
    logger.info("Validation passed")
    
    # Do work...
    result = perform_work(input_file, output_dir, logger)
    
    return {
        "output_file": result['path'],
        "items_processed": result['count'],
        "duration_seconds": result['time']
    }
```

### 3. Error Handling

✅ **DO:**
- Let exceptions propagate to be caught by TaskRunner
- Log errors with full context before raising
- Use specific exception types

```python
def robust_task(logger, config_file):
    try:
        logger.info(f"Loading config from {config_file}")
        config = load_config(config_file)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        raise ValueError(f"Config file is not valid JSON: {e}")
    
    try:
        logger.info("Starting processing...")
        result = process(config)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {type(e).__name__}: {e}")
        raise
```

### 4. Task Naming and Organization

✅ **DO:**
- Use descriptive task names
- Use consistent prefixes for task types
- Use meaningful task types for filtering

```python
# ✅ Good naming
task_id = task_manager.start(
    align_sequences,
    sequences_file,
    prefix="align-",
    task_name="Sequence Alignment: sample_123.fasta",
    task_type="alignment"
)

# ❌ Bad naming
task_id = task_manager.start(
    align_sequences,
    sequences_file,
    prefix="task-",  # Generic prefix
    task_name="Task",  # Not descriptive
    task_type="general"  # Not specific
)
```

### 5. Monitoring and Debugging

✅ **DO:**
- Check task status before assuming completion
- Read logs when tasks fail
- Implement proper error messages

```python
def monitor_task(task_id, timeout=300):
    """
    Monitor a task with timeout.
    
    Args:
        task_id: Task ID to monitor
        timeout: Maximum wait time in seconds
    
    Returns:
        Task result if successful
    
    Raises:
        TimeoutError: If task doesn't complete in time
        RuntimeError: If task fails
    """
    task_manager = get_task_manager()
    start_time = time.time()
    
    while True:
        logs, status, result = task_manager.poll(task_id)
        
        if "Done" in status:
            print(f"✅ Task completed successfully")
            return result
        
        if "Failed" in status:
            print(f"❌ Task failed. Recent logs:")
            print(logs[-1000:])
            raise RuntimeError(f"Task {task_id} failed")
        
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Task {task_id} timeout after {timeout}s")
        
        time.sleep(5)  # Poll every 5 seconds
```

---

## API Reference

### LogManager

#### `__init__(log_dir: str = "./logs", max_tail_bytes: int = 80_000)`
Initialize LogManager with specified directory and tail read size.

#### `new_session(prefix: str = "task-") -> str`
Create a new logging session and return unique session ID.

#### `get_logger(sid: str) -> logging.Logger`
Get logger instance for a session.

#### `get_log_path(sid: str) -> Path`
Get file path for a session's log file.

#### `read_tail(sid: str) -> str`
Read the tail of a log file (last N bytes).

---

### TaskRunner

#### `__init__(log_manager: LogManager)`
Initialize TaskRunner with a LogManager instance.

#### `start(task_fn: Callable, *args, prefix: str = "task-", task_name: str = "Unnamed Task", task_type: str = "general") -> str`
Start a background task.

**Parameters:**
- `task_fn`: Function to execute (must accept logger as first parameter)
- `*args`: Arguments to pass to task_fn
- `prefix`: Prefix for session ID (default: "task-")
- `task_name`: Human-readable task name
- `task_type`: Task category/type

**Returns:** Unique session ID

#### `poll(sid: str) -> Tuple[str, str, str]`
Poll task state and return (logs, status, result).

**Returns:**
- `logs`: Recent log content
- `status`: Task status emoji + text
- `result`: Task result (if completed)

#### `get_all_tasks() -> List[Dict[str, Any]]`
Get metadata for all tasks, sorted by start time (newest first).

#### `get_task_info(sid: str) -> Dict[str, Any]`
Get detailed information for a specific task.

---

### Global Task Manager

#### `get_task_manager() -> TaskRunner`
Get the global TaskRunner singleton instance.

#### `get_log_manager() -> LogManager`
Get the global LogManager singleton instance.

---

### Utility Functions

#### `run_command(cmd: str, logger) -> None`
Run a shell command and stream stdout/stderr to logger.

**Parameters:**
- `cmd`: Shell command to execute
- `logger`: Logger instance for output

**Raises:** `RuntimeError` if command fails

---

## Troubleshooting

### Common Issues

**Issue: Task appears stuck**
```python
# Solution: Check logs for errors
logs, status, _ = task_manager.poll(task_id)
print(f"Status: {status}")
print(f"Last 500 chars of logs:\n{logs[-500:]}")
```

**Issue: Log file not found**
```python
# Solution: Ensure task was started correctly
task_info = task_manager.get_task_info(task_id)
if not task_info:
    print(f"Task {task_id} does not exist")
else:
    log_path = log_manager.get_log_path(task_id)
    print(f"Log file location: {log_path}")
```

**Issue: Task function not receiving logger**
```python
# ❌ Wrong
def my_task(param1, param2):
    print("No logger!")

# ✅ Correct
def my_task(logger, param1, param2):
    logger.info("Has logger!")
```

---

## Performance Considerations

1. **Log File Size**: Logs rotate daily and keep 7 days of history
2. **Tail Reading**: Only last 80KB read by default for efficiency
3. **Thread Safety**: All operations are thread-safe
4. **Memory Usage**: Each task runs in separate daemon thread

---

## Version History

- **v1.0** (2025-10): Initial release with basic task and logging functionality
- **v1.1** (2025-10): Added metadata tracking and task monitoring
- **v1.2** (2025-10): Added global singleton pattern for application-wide access

---

## License

This system is part of SJTU-software-CASPIA project.

---

## Support

For questions or issues, please refer to the main project documentation or contact the development team.

