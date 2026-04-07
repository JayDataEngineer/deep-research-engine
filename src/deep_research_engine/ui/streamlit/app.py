"""Streamlit frontend for Deep Research Engine."""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from deep_research_engine.config import load_config
from deep_research_engine.engine import run_workflow


def inject_custom_css():
    st.markdown("""
        <style>
        /* Base Configuration & Fonts (Linear/Vercel inspired) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        code, pre, [class*="css"] code {
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
            font-size: 0.85rem !important;
        }

        /*
         * Layout adjustments to hide Streamlit default artifacts
         */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 3rem !important;
            max-width: 900px !important; /* Cap width for better readability */
        }

        /*
         * The Deep Dark Core Theme
         * Colors: Pure black/near-black backgrounds with subtle gray/white text,
         * accented with sharp borders and very subtle glowing effects.
         */
        .stApp {
            background-color: #000000;
            background-image: radial-gradient(circle at top right, rgba(45, 55, 72, 0.15) 0%, transparent 40%),
                              radial-gradient(circle at bottom left, rgba(66, 153, 225, 0.05) 0%, transparent 40%);
            color: #EDEDED;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            color: #FFFFFF !important;
        }

        h1 {
            font-size: 2.5rem !important;
            margin-bottom: 0.2rem !important;
        }

        p {
            color: #A1A1AA !important; /* Zinc 400 */
            line-height: 1.6 !important;
            font-size: 0.95rem !important;
        }

        /* Sidebar: Glassmorphism / Minimal */
        [data-testid="stSidebar"] {
            background-color: rgba(9, 9, 11, 0.6) !important; /* Zinc 950 with opacity */
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.85rem !important;
        }

        [data-testid="stSidebar"] h2 {
            color: #FFFFFF !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
            opacity: 0.6;
        }

        /* Forms & Inputs: Dark, crisp borders, subtle focus rings */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #09090B !important; /* Zinc 950 */
            border-radius: 6px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
        }

        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="input"] > div:hover {
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: #EDEDED !important;
            box-shadow: 0 0 0 1px #EDEDED !important;
        }

        div[data-baseweb="select"] * {
            color: #EDEDED !important;
        }

        /* Buttons: The primary interaction point */
        .stButton > button {
            background-color: #FAFAFA !important; /* Almost white primary button */
            color: #000000 !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        }

        .stButton > button:hover {
            background-color: #FFFFFF !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        }

        .stButton > button:active {
            transform: translateY(0) !important;
            background-color: #E5E5E5 !important;
        }

        /* Secondary Buttons (History items) */
        [data-testid="stSidebar"] .stButton > button {
            background-color: transparent !important;
            color: #A1A1AA !important;
            border: 1px solid transparent !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 0.5rem !important;
            font-weight: 400 !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            color: #FFFFFF !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(255, 255, 255, 0.05) !important;
            transform: none !important;
        }

        /* Main Chat Input: Floating, pill-like, sophisticated */
        [data-testid="stChatInput"] {
            background-color: #09090B !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px !important;
            padding: 0.2rem !important;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stChatInput"] textarea {
            color: #EDEDED !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.9), 0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
            transform: translateY(-2px) !important;
        }

        /* Chat Messages: Clean, borderless, distinguished by subtle background */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 1.5rem 0 !important; /* Just vertical spacing */
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stChatMessage"]:last-child {
            border-bottom: none !important;
        }

        [data-testid="stChatMessageContent"] {
            color: #D4D4D8 !important; /* Zinc 300 */
        }

        /* Remove background distinction for user/assistant, rely on avatar */
        [data-testid="stChatMessage"][data-testid*="user"] {
            background-color: transparent !important;
        }

        /* Avatars */
        [data-testid="stChatMessageAvatarUser"] {
            background-color: #27272A !important;
            color: #EDEDED !important;
            border-radius: 4px !important;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #FAFAFA !important;
            color: #000000 !important;
            border-radius: 4px !important;
        }

        /* Status & Expanders: Minimal line design */
        [data-testid="stStatusWidget"] {
            background-color: #09090B !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
        }

        /* File Uploader: Dotted minimal box */
        [data-testid="stFileUploader"] {
            background-color: #000000 !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
            border: 1px dashed rgba(255, 255, 255, 0.2) !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(255, 255, 255, 0.5) !important;
            background-color: rgba(255, 255, 255, 0.02) !important;
        }

        /* Dividers */
        hr {
            border-color: rgba(255, 255, 255, 0.08) !important;
            margin: 2rem 0 !important;
        }

        /* Dataframes & Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }

        /* Markdown Links */
        a {
            color: #EDEDED !important;
            text-decoration: underline !important;
            text-underline-offset: 4px !important;
            text-decoration-color: rgba(255, 255, 255, 0.3) !important;
            transition: all 0.2s ease !important;
        }

        a:hover {
            text-decoration-color: #EDEDED !important;
        }

        /* Scrollbars (Webkit) */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)


def setup_page():
    """Setup streamlit page configuration."""
    st.set_page_config(
        page_title="Deep Research Engine",
        page_icon="❖",
        layout="centered", # Centered layout is often more readable for chat/docs
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Minimalist Header
    st.markdown("<h1>❖ Deep Research Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: \"JetBrains Mono\", monospace; font-size: 0.8rem !important; letter-spacing: 0.05em; color: #52525B !important; margin-top: -5px; margin-bottom: 2rem;'>VERSION 0.2.0-ALPHA // CORE SYSTEM</p>", unsafe_allow_html=True)


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
        st.error(f"Configuration file not found: {config_path}")
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
    with st.chat_message("assistant", avatar="❖"):
        status_placeholder = st.empty()

        # Minimal status styling
        with status_placeholder.status("Processing inquiry...", expanded=True) as status:
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
                        status.write(f"› Initiating: **{event['phase']}**")
                        st.session_state.current_status = f"Running: {event['phase']}"
                    elif event_type == "phase_complete":
                        status.write(f"✓ Completed: **{event['phase']}**")
                    elif event_type == "worker_complete":
                         status.write(f"  ↳ Worker **{event.get('worker_id', 'unknown')}** finished")
                    elif event_type == "artifact_maker_complete":
                        status.write("✦ Artifact synthesized successfully")
                        if event.get("output"):
                            full_response = event["output"]
                            response_placeholder.markdown(full_response)
                    elif event_type == "complete":
                        status.update(label="Process complete", state="complete", expanded=False)
                        if event.get("output") and not full_response:
                            full_response = event["output"]
                            response_placeholder.markdown(full_response)
                    elif event_type == "error":
                        st.error(f"Error encountered: {event.get('error')}")
                        status.update(label="Process failed", state="error")
                
                # Save the full response to session state
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    msg = "Process concluded without generating an artifact."
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    response_placeholder.info(msg)
                     
            except Exception as e:
                st.error(f"System Exception: {e}")
                status.update(label="System Exception", state="error")
            finally:
                st.session_state.is_running = False


def main():
    """Main Streamlit entry point."""
    setup_page()
    init_session_state()

    # Sidebar layout (Minimalist)
    with st.sidebar:
        # App identity in sidebar top
        st.markdown("<div style='display:flex; align-items:center; gap: 8px; margin-bottom: 2rem;'><h1 style='font-size: 1.5rem !important; margin: 0 !important;'>❖ DRE</h1></div>", unsafe_allow_html=True)

        st.markdown("<h2>Workspace Settings</h2>", unsafe_allow_html=True)
        config_options = ["rag.yaml", "chat.yaml", "pdf_summarizer.yaml", "telegram_ingest.yaml", "skill_generator.yaml"]
        selected_config = st.selectbox("Workflow Pipeline", options=config_options, index=0)
        
        thread_id = st.text_input("Session Identifier", value=st.session_state.session_id)
        if thread_id != st.session_state.session_id:
            st.session_state.session_id = thread_id
            st.session_state.messages = []
            
        st.markdown("---")

        st.markdown("<h2>Data Sources</h2>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg", "txt"], label_visibility="collapsed")
        
        st.markdown("---")

        st.markdown("<h2>Recent Activity</h2>", unsafe_allow_html=True)
        history = get_history()

        if not history:
            st.markdown("<p style='font-size: 0.8rem;'>No recent activity.</p>", unsafe_allow_html=True)

        for session in history:
            # Clean layout for history items
            if st.button(f"› {session['query']}", key=f"btn_{session['id']}", use_container_width=True):
                st.session_state.session_id = session['id']
                st.session_state.messages = [{"role": "user", "content": session.get("full_query", session["query"])}]
                st.rerun()
            
        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()

    # Display chat history in the main container
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "❖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question, request an artifact, or provide commands..."):
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
            st.warning("System is currently processing. Please wait.")
    
    # If no prompt but files uploaded, show a contextual start button in the main area
    if not prompt and uploaded_files and not st.session_state.is_running and len(st.session_state.messages) == 0:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info(f"**{len(uploaded_files)} file(s)** ready for processing via `{selected_config}` pipeline.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Execute Data Pipeline", use_container_width=True):
                 try:
                    loop = asyncio.get_event_loop()
                 except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                 loop.run_until_complete(run_research("Process uploaded knowledge sources", selected_config, st.session_state.session_id, uploaded_files))
                 st.rerun()


if __name__ == "__main__":
    main()
