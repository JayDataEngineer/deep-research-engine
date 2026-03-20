import asyncio
import os
from pathlib import Path
from deep_research_engine import run_workflow, load_config

async def main():
    # 1. Load the new social archive config
    config_path = "configs/social_archive.yaml"
    
    # Set the EXPORT_PATH environment variable
    os.environ["EXPORT_PATH"] = "data/ChatExport_2026-03-13"
    
    config = load_config(config_path)
    
    # 2. Run the workflow
    # We use a session ID to track it
    session_id = "social_archive_session"
    print(f"Starting Social Archive Workflow for: {os.environ['EXPORT_PATH']}")
    print("-" * 50)
    
    async for event in run_workflow(
        input_data={"source": os.environ["EXPORT_PATH"], "query": "Who are my friends and what is our history?"},
        config=config,
        session_id=session_id,
        stream=True
    ):
        event_type = event.get("type")
        if event_type == "phase_start":
            print(f"\n>>> Phase: {event.get('phase').upper()}")
        elif event_type == "phase_complete":
            print(f"<<< Phase {event.get('phase').upper()} Complete")
        elif event_type == "phase_retry":
            print(f"!!! RETRYING PHASE: {event.get('phase')} ({event.get('reason')})")
        elif event_type == "error":
            print(f"!!! ERROR: {event.get('error')} in phase {event.get('phase')}")
        elif event_type == "complete":
            print("\n" + "=" * 50)
            print("WORKFLOW COMPLETE")
            print("=" * 50)
            print("\nFinal Synthesized Output:")
            print(event.get("output"))
            
            # Check for the big text dump
            state = event.get("state", {})
            big_dump = state.get("context", {}).get("big_text_dump")
            if big_dump:
                print(f"\nBig Text Dump saved at: {big_dump}")

if __name__ == "__main__":
    asyncio.run(main())
