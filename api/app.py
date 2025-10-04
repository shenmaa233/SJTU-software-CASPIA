# api/app.py
import json
from fastapi import FastAPI, WebSocket, UploadFile, WebSocketDisconnect
from src.CASPIAgent.service import AgentService

app = FastAPI()
agent_service = AgentService()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收前端消息
            data = await websocket.receive_text()
            payload = json.loads(data)

            message = payload.get("message", "")
            history = payload.get("history", [])
            tool_history = payload.get("tool_history", "")
            session_state = payload.get("session_state", {})
            uploaded_file = None  # 这里可以拓展成 UploadFile

            # 构建 Agent
            agent_executor, final_input, chat_history, session_state = await agent_service.run(
                message, history, tool_history, uploaded_file, session_state
            )

            # 流式推理
            async for event in agent_executor.astream_events(
                {"input": final_input, "chat_history": chat_history}, version="v1"
            ):
                event_name = event["event"]
                if event_name == "on_tool_start":
                    await websocket.send_text(json.dumps({
                        "type": "tool_start",
                        "tool_name": event["name"]
                    }))
                elif event_name == "on_tool_end":
                    await websocket.send_text(json.dumps({
                        "type": "tool_end",
                        "tool_name": event["name"],
                        "output": event["data"].get("output", "")
                    }))
                elif event_name == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        await websocket.send_text(json.dumps({
                            "type": "token",
                            "content": chunk
                        }))

            await websocket.send_text(json.dumps({
                "type": "done",
                "session_state": session_state
            }))
    except WebSocketDisconnect:
        print("WebSocket disconnected")
