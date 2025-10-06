"""
========================================
CASPIAgent Tab

Author: SJTU-Software Team
Date: 2025-10-04
========================================
"""

import gradio as gr
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from src.CASPIAgent.service import AgentService


# ========================================
# Configuration
# ========================================

# Agent service
agent_service = AgentService()

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ========================================
# File upload handler
# ========================================

def handle_file_upload(file):
    """
    Handle file upload (customizable interface function)
    
    Args:
        file: Gradio File object (temporary path)
        
    Returns:
        tuple: (saved file path, display message)
    """
    if file is None:
        return None, "No file selected"
    
    try:
        # Get original file name
        original_name = Path(file.name).name
        
        # Generate timestamped file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(original_name).suffix
        new_filename = f"{timestamp}_{original_name}"
        
        # Save path
        saved_path = UPLOAD_DIR / new_filename
        
        # Copy file
        shutil.copy2(file.name, saved_path)
        
        message = f"✅ File uploaded: {original_name}\nSaved at: {saved_path}"
        
        return str(saved_path), message
        
    except Exception as e:
        return None, f"❌ Upload failed: {str(e)}"


# ========================================
# Chat handler
# ========================================

async def chat_with_agent(message, history, uploaded_file_path, session_state):
    """
    Chat with Agent (streaming generation)
    
    Args:
        message: User message
        history: Chat history [[user_msg, bot_msg], ...]
        uploaded_file_path: Uploaded file path
        session_state: Session state
        
    Yields:
        tuple: (history, tool call text, session state)
    """
    if not message.strip():
        yield history, "", session_state
        return
    
    # Prepare file object
    file_obj = None
    if uploaded_file_path:
        class FileWrapper:
            def __init__(self, path):
                self.name = path
        file_obj = FileWrapper(uploaded_file_path)
    
    # Convert history format
    history_dict = []
    for user_msg, bot_msg in history:
        if user_msg:
            history_dict.append({"role": "user", "content": user_msg})
        if bot_msg:
            history_dict.append({"role": "assistant", "content": bot_msg})
    
    try:
        # Build agent
        agent_executor, final_input, chat_history, new_session_state = await agent_service.run(
            message, history_dict, file_obj, session_state
        )
        
        # Add user message
        history.append([message, ""])
        
        # Tool call log
        tool_calls_text = ""
        current_response = ""
        
        # Streaming execution
        async for event in agent_executor.astream_events(
            {"input": final_input, "chat_history": chat_history},
            version="v1"
        ):
            event_name = event["event"]
            
            # Tool start
            if event_name == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_calls_text += f"🔧 Tool called: {tool_name}\n"
                yield history, tool_calls_text, session_state
            
            # Tool end
            elif event_name == "on_tool_end":
                tool_name = event.get("name", "unknown")
                output = str(event["data"].get("output", ""))[:200]
                tool_calls_text += f"✅ {tool_name} finished\n{output}\n\n"
                yield history, tool_calls_text, session_state
            
            # Streaming content
            elif event_name == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        current_response += content
                        history[-1][1] = current_response
                        yield history, tool_calls_text, session_state
            
            # Chain end
            elif event_name == "on_chain_end":
                output = event["data"].get("output")
                if output and isinstance(output, dict) and "output" in output:
                    if not current_response:
                        current_response = output["output"]
                        history[-1][1] = current_response
        
        # Ensure response
        if not history[-1][1]:
            history[-1][1] = "Completed."
        
        yield history, tool_calls_text, new_session_state
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        if history and len(history) > 0:
            history[-1][1] = error_msg
        else:
            history.append([message, error_msg])
        yield history, tool_calls_text, session_state


# ========================================
# UI
# ========================================

def agent_tab():
    """Create Agent Tab"""
    
    # State variables
    session_state = gr.State(dict)
    uploaded_file_path = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=2):
            
            # Chat window
            with gr.Blocks():
                
                # Title
                gr.Markdown("## 🤖 CASPIAgent")
                gr.Markdown("An intelligent assistant for synthetic biology and metabolic engineering")

                chatbot = gr.Chatbot(
                    label="Chat",
                    value=[],  # Initialize with empty list to ensure rendering
                    height=500,
                    show_copy_button=True,
                    container=True,
                    show_label=True,
                    placeholder="💬 Start a conversation by typing your question below..."
                )
            
            # Input area
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Enter your question...",
                    scale=4,
                    lines=2,
                )
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="sm")
        
        with gr.Column(scale=1):
            # File upload
            gr.Markdown("### 📎 File Upload")
            file_input = gr.File(
                label="Select File",
                file_types=None,  # Accept all file types
            )
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                lines=3,
            )
            
            # Tool call display
            gr.Markdown("### 🔧 Tool Calls")
            tool_display = gr.Textbox(
                label="",
                interactive=False,
                lines=10,
                show_label=False,
            )
            
            # Quick examples
            gr.Markdown("### 💡 Quick Examples")
            example_btn1 = gr.Button("📝 Genome Annotation", size="sm")
            example_btn2 = gr.Button("🔬 Kcat Prediction", size="sm")
            example_btn3 = gr.Button("❓ Feature Introduction", size="sm")
    
    # User guide
    with gr.Accordion("📖 User Guide", open=False):
        gr.Markdown("""
        ### Features
        
        1. **File Upload**: Click "Select File" on the right to upload any type of file
        2. **Chat**: Enter your question in the input box and click Send
        3. **Tool Calls**: The agent will automatically call the appropriate tool, and the execution status will be displayed on the right
        4. **Quick Examples**: Click the quick buttons for common questions
        
        ### Available Tools
        
        - 🧬 Gene Prediction (GeneMarkS)
        - 🔬 Kcat Prediction
        - 📊 Protein Extraction
        """)
    
    # ========================================
    # Event bindings
    # ========================================
    
    # File upload
    file_input.change(
        fn=handle_file_upload,
        inputs=[file_input],
        outputs=[uploaded_file_path, upload_status],
    )
    
    # Send message
    send_event = send_btn.click(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot, uploaded_file_path, session_state],
        outputs=[chatbot, tool_display, session_state],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    )
    
    # Enter to send
    msg_input.submit(
        fn=chat_with_agent,
        inputs=[msg_input, chatbot, uploaded_file_path, session_state],
        outputs=[chatbot, tool_display, session_state],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    )
    
    # Clear chat
    clear_btn.click(
        fn=lambda: ([], "", {}),
        outputs=[chatbot, tool_display, session_state],
    )
    
    # Quick examples
    example_btn1.click(
        fn=lambda: "Please analyze and annotate the uploaded file.",
        outputs=[msg_input],
    )
    
    example_btn2.click(
        fn=lambda: "Please help me predict the kcat value of an enzyme.",
        outputs=[msg_input],
    )
    
    example_btn3.click(
        fn=lambda: "What features do you have? What can you help me with?",
        outputs=[msg_input],
    )


# ========================================
# Test entry
# ========================================

if __name__ == "__main__":
    """Standalone test"""
    with gr.Blocks(title="CASPIAgent - Simple") as demo:
        agent_tab()
    
    demo.launch(server_name="0.0.0.0", share=False, debug=True)
