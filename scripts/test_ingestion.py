import asyncio
import os
from pathlib import Path
from deep_research_engine.nodes.ingestion import ingestion_node
from deep_research_engine.state.types import create_initial_state

async def main():
    # Setup state
    session_id = "test_telegram_archive"
    input_path = "data/ChatExport_2026-03-13"
    
    state = create_initial_state(
        workflow_name="telegram_archive",
        workflow_type="ingestion",
        input_data={"path": input_path},
        session_id=session_id,
        run_id="run_1"
    )
    
    # Run ingestion node
    print(f"Starting ingestion for {input_path}...")
    final_state = await ingestion_node(state)
    
    print("\nIngestion Results:")
    for result in final_state.get("ingestion_results", []):
        print(f" - {result}")
        
    big_dump = final_state.get("context", {}).get("big_text_dump")
    print(f"\nBig Text Dump created at: {big_dump}")
    
    if big_dump and Path(big_dump).exists():
        print("\nFirst 20 lines of big dump:")
        with open(big_dump, "r") as f:
            for _ in range(20):
                print(f.readline().strip())

if __name__ == "__main__":
    asyncio.run(main())
