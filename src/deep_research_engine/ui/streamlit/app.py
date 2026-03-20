"""Streamlit frontend for Deep Research Engine."""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from deep_research_engine.config import load_config
from deep_research_engine.engine import run_workflow


def setup_page():
    """Setup streamlit page configuration."""
    st.set_page_config(
        page_title="Deep Research Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🔍 Deep Research Engine")
    st.markdown("---")


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_status" not in st.session_state:
        st.session_state.current_status = ""


def save_session_to_history(session_id: str, query: str, config_name: str):
    """Save session metadata to a local history file."""
    history_file = Path("data/ui/history.json")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    history = {"sessions": []}
    if history_file.exists():
        try:
            with history_file.open() as f:
                history = json.load(f)
        except Exception:
            pass
    
    # Check if already exists
    for s in history["sessions"]:
        if s["id"] == session_id:
            return
            
    history["sessions"].insert(0, {
        "id": session_id,
        "query": query[:50] + ("..." if len(query) > 50 else ""),
        "full_query": query,
        "config": config_name,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep last 20
    history["sessions"] = history["sessions"][:20]
    
    with history_file.open("w") as f:
        json.dump(history, f, indent=2)


def get_history():
    """Get previous sessions from history file."""
    history_file = Path("data/ui/history.json")
    if not history_file.exists():
        return []
    try:
        with history_file.open() as f:
            return json.load(f).get("sessions", [])
    except Exception:
        return []


async def run_research(query: str, config_name: str, thread_id: str, uploaded_files: list = None):
    """Run research workflow and stream results."""
    # Load config
    configs_dir = Path(__file__).parent.parent.parent.parent / "configs"
    config_path = configs_dir / config_name
    
    if not config_path.exists():
        st.error(f"Config not found: {config_path}")
        return

    # Handle uploaded files
    input_data = query
    if uploaded_files:
        upload_dir = Path("data/uploads") / thread_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = upload_dir / uploaded_file.name
            with file_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))
        
        # If it's a PDF workflow, use the first PDF as PDF_PATH
        if config_name == "pdf_summarizer.yaml" and file_paths:
            os.environ["PDF_PATH"] = file_paths[0]
            input_data = {"pdf_path": file_paths[0], "query": query}
        else:
            input_data = {"query": query, "files": file_paths}

    config = load_config(str(config_path))
    
    # Save to history
    save_session_to_history(thread_id, query, config_name)
    
    # Store message in state first
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.is_running = True
    
    # Create containers for the UI
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        # Use a status component for phases
        with status_placeholder.status("🔍 Researching...", expanded=True) as status:
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                async for event in run_workflow(
                    input_data,
                    config,
                    session_id=thread_id,
                    stream=True,
                ):
                    event_type = event.get("type", "unknown")
                    
                    if event_type == "phase_start":
                        status.write(f"📍 Starting phase: **{event['phase']}**")
                        st.session_state.current_status = f"Running: {event['phase']}"
                    elif event_type == "phase_complete":
                        status.write(f"✅ Completed phase: **{event['phase']}**")
                    elif event_type == "worker_complete":
                         status.write(f"  🛠️ Worker **{event.get('worker_id', 'unknown')}** complete")
                    elif event_type == "synthesis_complete":
                        status.write("✨ Synthesis complete!")
                        if event.get("output"):
                            full_response = event["output"]
                            response_placeholder.markdown(full_response)
                    elif event_type == "complete":
                        status.update(label="✅ Research Complete!", state="complete", expanded=False)
                        if event.get("output") and not full_response:
                            full_response = event["output"]
                            response_placeholder.markdown(full_response)
                    elif event_type == "error":
                        st.error(f"❌ Error: {event.get('error')}")
                        status.update(label="❌ Research Failed", state="error")
                
                # Save the full response to session state
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    msg = "Workflow finished but no output was generated."
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    response_placeholder.info(msg)
                     
            except Exception as e:
                st.error(f"Critical Error: {e}")
                status.update(label="❌ Critical Error", state="error")
            finally:
                st.session_state.is_running = False


def main():
    """Main Streamlit entry point."""
    setup_page()
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        config_options = ["rag.yaml", "chat.yaml", "pdf_summarizer.yaml", "telegram_ingest.yaml", "skill_generator.yaml"]
        selected_config = st.selectbox("Workflow Config", options=config_options, index=0)
        
        thread_id = st.text_input("Thread ID (for persistence)", value=st.session_state.session_id)
        if thread_id != st.session_state.session_id:
            st.session_state.session_id = thread_id
            # Reset messages when switching thread
            st.session_state.messages = []
            
        st.markdown("---")
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF or Images", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg", "txt"])
        
        st.markdown("---")
        st.header("📜 History")
        history = get_history()
        for session in history:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"{session['query']}", key=f"btn_{session['id']}", use_container_width=True):
                    st.session_state.session_id = session['id']
                    # In a real app, we'd load messages from DB, here we just reset for simplicity or display metadata
                    st.session_state.messages = [{"role": "user", "content": session.get("full_query", session["query"])}]
                    st.rerun()
            with col2:
                 st.caption(f"{session['config'].split('.')[0]}")
            
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
            
        add_vertical_space(5)
        st.caption("Deep Research Engine v0.2.0")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question or request an artifact..."):
        if not st.session_state.is_running:
            # We use an event loop to run the async function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(run_research(prompt, selected_config, st.session_state.session_id, uploaded_files))
            st.rerun()
        else:
            st.warning("A research task is already running. Please wait.")
    
    # If no prompt but files uploaded, show a "Start" button
    if not prompt and uploaded_files and not st.session_state.is_running:
        if st.button("🚀 Start Pipeline with Uploaded Files"):
             try:
                loop = asyncio.get_event_loop()
             except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
             
             loop.run_until_complete(run_research("Processes uploaded files", selected_config, st.session_state.session_id, uploaded_files))
             st.rerun()


if __name__ == "__main__":
    main()
