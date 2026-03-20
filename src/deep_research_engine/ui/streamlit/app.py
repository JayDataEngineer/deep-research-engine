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


def inject_custom_css():
    st.markdown("""
        <style>
        /* General Theme Settings */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #0B0F19;
            background-image: radial-gradient(circle at 50% 0%, #1E293B 0%, transparent 50%);
            color: #E2E8F0;
        }

        /* Typography & Gradients */
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }

        h1 {
            background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        [data-testid="stSidebar"] h2 {
            color: #94A3B8 !important;
            text-transform: uppercase;
            font-size: 0.85rem !important;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
            color: #E2E8F0 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(129, 140, 248, 0.5) !important;
            background: linear-gradient(180deg, rgba(49, 46, 129, 0.8) 0%, rgba(30, 27, 75, 0.8) 100%) !important;
        }

        /* Chat Input */
        [data-testid="stChatInput"] {
            border-radius: 16px !important;
            background-color: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(12px) !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3) !important;
        }

        [data-testid="stChatInput"] textarea {
            color: #F8FAFC !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #818CF8 !important;
            box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2), 0 20px 25px -5px rgba(0, 0, 0, 0.3) !important;
        }

        /* Chat Messages */
        [data-testid="stChatMessage"] {
            background-color: rgba(30, 41, 59, 0.4) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(8px) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }

        /* User message distinct style */
        [data-testid="stChatMessage"][data-testid*="user"] {
            background-color: rgba(99, 102, 241, 0.1) !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
        }

        /* Status Expanders / Status */
        [data-testid="stStatusWidget"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(8px) !important;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: rgba(30, 41, 59, 0.4) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: 1px dashed rgba(129, 140, 248, 0.5) !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stFileUploader"]:hover {
            background-color: rgba(30, 41, 59, 0.6) !important;
            border-color: rgba(129, 140, 248, 0.8) !important;
        }

        /* Info/Warning/Error alerts */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }

        /* Selectbox */
        div[data-baseweb="select"] > div {
            background-color: rgba(30, 41, 59, 0.6) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        /* Text input */
        div[data-baseweb="input"] > div {
            background-color: rgba(30, 41, 59, 0.6) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        /* Dividers */
        hr {
            border-color: rgba(255, 255, 255, 0.05) !important;
        }

        /* Top bar padding fix */
        .block-container {
            padding-top: 2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)


def setup_page():
    """Setup streamlit page configuration."""
    st.set_page_config(
        page_title="Deep Research Engine",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.markdown("<h1>Deep Research Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #94A3B8; margin-top: -10px; margin-bottom: 2rem;'>Next-Generation Autonomous Intelligence Platform</p>", unsafe_allow_html=True)


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
    with st.chat_message("assistant", avatar="🔮"):
        status_placeholder = st.empty()
        # Use a status component for phases
        with status_placeholder.status("🧠 Initializing Deep Research Engine...", expanded=True) as status:
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
                        status.write(f"🔄 **Phase started:** {event['phase']}")
                        st.session_state.current_status = f"Running: {event['phase']}"
                    elif event_type == "phase_complete":
                        status.write(f"✅ **Phase completed:** {event['phase']}")
                    elif event_type == "worker_complete":
                         status.write(f"  ⚡ **Worker resolved:** {event.get('worker_id', 'unknown')}")
                    elif event_type == "artifact_maker_complete":
                        status.write("✨ **Artifact synthesis complete!**")
                        if event.get("output"):
                            full_response = event["output"]
                            response_placeholder.markdown(full_response)
                    elif event_type == "complete":
                        status.update(label="✅ **Mission Accomplished!**", state="complete", expanded=False)
                        if event.get("output") and not full_response:
                            full_response = event["output"]
                            response_placeholder.markdown(full_response)
                    elif event_type == "error":
                        st.error(f"❌ Critical Error: {event.get('error')}")
                        status.update(label="❌ Mission Failed", state="error")
                
                # Save the full response to session state
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    msg = "Mission concluded but no direct artifact was generated."
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    response_placeholder.info(msg)
                     
            except Exception as e:
                st.error(f"System Failure: {e}")
                status.update(label="❌ System Failure", state="error")
            finally:
                st.session_state.is_running = False


def main():
    """Main Streamlit entry point."""
    setup_page()
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration Profile")
        config_options = ["rag.yaml", "chat.yaml", "pdf_summarizer.yaml", "telegram_ingest.yaml", "skill_generator.yaml"]
        selected_config = st.selectbox("Intelligence Protocol", options=config_options, index=0)
        
        thread_id = st.text_input("Thread ID (Persistence Layer)", value=st.session_state.session_id)
        if thread_id != st.session_state.session_id:
            st.session_state.session_id = thread_id
            # Reset messages when switching thread
            st.session_state.messages = []
            
        st.markdown("---")
        st.header("📁 Knowledge Ingestion")
        uploaded_files = st.file_uploader("Drop Knowledge Sources", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg", "txt"])
        
        st.markdown("---")
        st.header("📜 Operation History")
        history = get_history()
        for session in history:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"🔍 {session['query']}", key=f"btn_{session['id']}", use_container_width=True):
                    st.session_state.session_id = session['id']
                    # In a real app, we'd load messages from DB, here we just reset for simplicity or display metadata
                    st.session_state.messages = [{"role": "user", "content": session.get("full_query", session["query"])}]
                    st.rerun()
            with col2:
                 st.caption(f"{session['config'].split('.')[0]}")
            
        st.markdown("---")
        if st.button("🗑️ Purge Current Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
            
        add_vertical_space(3)
        st.markdown("<div style='text-align: center; color: #475569; font-size: 0.8rem;'>DRE v0.2.0-alpha<br>Secured Channel</div>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🔮"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter command or intelligence query..."):
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
            st.warning("⚠️ An operation is currently in progress. Please hold.")
    
    # If no prompt but files uploaded, show a "Start" button
    if not prompt and uploaded_files and not st.session_state.is_running:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Execute Pipeline on Ingested Data", use_container_width=True):
             try:
                loop = asyncio.get_event_loop()
             except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
             
             loop.run_until_complete(run_research("Processes uploaded files", selected_config, st.session_state.session_id, uploaded_files))
             st.rerun()


if __name__ == "__main__":
    main()
