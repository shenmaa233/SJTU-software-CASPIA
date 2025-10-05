# tabs/tasks_monitor_tab.py

import gradio as gr
from src.utils import get_task_manager
from datetime import datetime


def format_task_list(tasks):
    """
    Format task list for display in dataframe.
    
    Args:
        tasks: List of task dictionaries
        
    Returns:
        List of lists for dataframe display
    """
    if not tasks:
        return [["No tasks yet", "", "", "", "", ""]]
    
    formatted = []
    for task in tasks:
        # Format timestamps
        start_time = task["start_time"]
        if isinstance(start_time, datetime):
            start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        end_time = task["end_time"]
        if isinstance(end_time, datetime):
            end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        elif end_time == "N/A":
            end_time = "Running..."
        
        # Calculate duration if task is done
        duration = "N/A"
        if task["end_time"] != "N/A" and isinstance(task["start_time"], datetime) and isinstance(task["end_time"], datetime):
            delta = task["end_time"] - task["start_time"]
            minutes = int(delta.total_seconds() // 60)
            seconds = int(delta.total_seconds() % 60)
            duration = f"{minutes}m {seconds}s"
        
        formatted.append([
            task["sid"],
            task["name"],
            task["type"],
            task["status"],
            start_time,
            duration
        ])
    
    return formatted


def refresh_task_list():
    """
    Refresh and return the current task list.
    
    Returns:
        Formatted task list for display
    """
    runner = get_task_manager()
    tasks = runner.get_all_tasks()
    return format_task_list(tasks)


def view_task_details(selected_data: gr.SelectData):
    """
    Show detailed logs for a selected task.
    
    Args:
        selected_data: Gradio SelectData object
        
    Returns:
        Tuple of (logs, status, result)
    """
    if selected_data is None:
        return "Please select a task from the table above.", "", ""
    
    # Get the task ID from the first column
    task_id = selected_data.value
    
    runner = get_task_manager()
    logs, status, result = runner.poll(task_id)
    
    if not logs:
        logs = "No logs available for this task."
    
    return logs, status, result


def auto_refresh_logs(task_id_input: str):
    """
    Auto-refresh logs for a specific task.
    
    Args:
        task_id_input: Task ID to monitor
        
    Returns:
        Tuple of (logs, status, result)
    """
    if not task_id_input:
        return "", "", ""
    
    runner = get_task_manager()
    return runner.poll(task_id_input)


def tasks_monitor_tab():
    """
    Create the Tasks Monitor tab UI.
    """
    gr.Markdown("""
    ## 📊 Tasks Monitor
    
    Track all running and completed tasks across the system. 
    Click on any task to view detailed logs and progress.
    """)
    
    # Task ID input for monitoring
    with gr.Row():
        task_id_input = gr.Textbox(
            label="Task ID (for monitoring specific task)",
            placeholder="Enter task ID or click a row in the table below",
            scale=4
        )
        refresh_btn = gr.Button("🔄 Refresh List", scale=1)
    
    # Task list table
    task_table = gr.Dataframe(
        headers=["Task ID", "Task Name", "Type", "Status", "Start Time", "Duration"],
        datatype=["str", "str", "str", "str", "str", "str"],
        label="All Tasks",
        interactive=False,
        wrap=True,
        value=refresh_task_list()
    )
    
    # Task details section
    gr.Markdown("### Task Details")
    
    with gr.Row():
        status_box = gr.Textbox(label="Status", interactive=False, scale=1)
        result_box = gr.Textbox(label="Result", interactive=False, scale=2)
    
    logs_box = gr.Textbox(
        label="Task Logs",
        lines=20,
        interactive=False,
        max_lines=30
    )
    
    # Event handlers
    refresh_btn.click(
        fn=refresh_task_list,
        outputs=[task_table]
    )
    
    # When user clicks a row in the table, fill in the task ID and show logs
    task_table.select(
        fn=view_task_details,
        outputs=[logs_box, status_box, result_box]
    )
    
    # Also update task_id_input when clicking a row
    def update_task_id(selected_data: gr.SelectData):
        return selected_data.value
    
    task_table.select(
        fn=update_task_id,
        outputs=[task_id_input]
    )
    
    # Auto-refresh logs when task_id changes
    task_id_input.change(
        fn=auto_refresh_logs,
        inputs=[task_id_input],
        outputs=[logs_box, status_box, result_box]
    )
    
    # Auto-refresh timer (updates every 2 seconds)
    # Note: Timers are now properly scoped within the tab
    timer = gr.Timer(2.0)
    timer.tick(
        fn=lambda task_id: auto_refresh_logs(task_id) if task_id else ("", "", ""),
        inputs=[task_id_input],
        outputs=[logs_box, status_box, result_box]
    )
    
    # Also refresh the task list every 5 seconds
    list_timer = gr.Timer(5.0)
    list_timer.tick(
        fn=refresh_task_list,
        outputs=[task_table]
    )

