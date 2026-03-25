import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from src.deep_research_engine.shared.multimodal import (
    extract_scene_frames,
    create_mega_files,
    unstitch_transcript,
    unify_speakers
)

@pytest.mark.asyncio
async def test_extract_scene_frames_parsing():
    # Mock metadata file content
    metadata_content = """
frame:0    pts:0       pts_time:0.0
lavfi.scene_score=0.450000
frame:1    pts:1536    pts_time:0.032
lavfi.scene_score=0.120000
frame:2    pts:3072    pts_time:0.064
lavfi.scene_score=0.510000
"""
    
    video_path = Path("fake_video.mp4")
    output_dir = Path("fake_output")
    
    with patch("pathlib.Path.mkdir"), \
         patch("asyncio.create_subprocess_exec") as mock_exec, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", pytest.raises(Exception)), \
         patch("pathlib.Path.glob", return_value=[Path("fake_output/frame_0001.png"), Path("fake_output/frame_0002.png")]):
        
        # We need a more sophisticated mock for open and glob
        pass

    # Actually, let's just test the regex logic directly if possible, or mock better
    with patch("src.deep_research_engine.shared.multimodal.asyncio.create_subprocess_exec") as mock_exec:
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0
        mock_exec.return_value = mock_process
        
        with patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: MagicMock(read=lambda: metadata_content)))), \
             patch("pathlib.Path.glob", return_value=[Path("fake_output/frame_0001.png"), Path("fake_output/frame_0002.png")]):
            
            frames = await extract_scene_frames(video_path, output_dir)
            
            assert len(frames) == 2
            assert frames[0]["pts_time"] == 0.0
            assert frames[0]["scene_score"] == 0.45
            assert frames[1]["pts_time"] == 0.032
            assert frames[1]["scene_score"] == 0.12

@pytest.mark.asyncio
async def test_create_mega_files_batching():
    audio_paths = [Path(f"audio_{i}.wav") for i in range(5)]
    durations = {str(Path(f"audio_{i}.wav")): 2000.0 for i in range(5)} # 2000s each
    
    # max_duration = 3600s (1hr)
    # 2000 + 2000 > 3600 -> should create batches of 1 file each if each is 2000s
    
    with patch("src.deep_research_engine.shared.multimodal.get_audio_duration", side_effect=lambda p: durations[str(p)]), \
         patch("src.deep_research_engine.shared.multimodal.asyncio.create_subprocess_exec") as mock_exec, \
         patch("pathlib.Path.mkdir"), \
         patch("builtins.open", MagicMock()):
        
        mock_process = MagicMock()
        mock_process.wait = AsyncMock()
        mock_process.returncode = 0
        mock_exec.return_value = mock_process
        
        boundary_maps = await create_mega_files(audio_paths, Path("output"), max_duration_sec=3600.0)
        
        assert len(boundary_maps) == 5
        for i, bm in enumerate(boundary_maps):
            assert len(bm["files"]) == 1
            assert bm["files"][0]["original_path"] == str(audio_paths[i])

@pytest.mark.asyncio
async def test_unstitch_transcript_mapping():
    asr_results = {
        "segments": [
            {"start": 10.0, "end": 20.0, "text": "Hello", "speaker": "S1"},
            {"start": 3700.0, "end": 3710.0, "text": "World", "speaker": "S2"}
        ]
    }
    boundary_map = {
        "files": [
            {"original_path": "/path/to/a.wav", "name": "a.wav", "start": 0.0, "end": 3600.0},
            {"original_path": "/path/to/b.wav", "name": "b.wav", "start": 3600.0, "end": 7200.0}
        ]
    }
    
    unstitched = unstitch_transcript(asr_results, boundary_map)
    
    assert "/path/to/a.wav" in unstitched
    assert "/path/to/b.wav" in unstitched
    assert len(unstitched["/path/to/a.wav"]) == 1
    assert unstitched["/path/to/a.wav"][0]["text"] == "Hello"
    assert unstitched["/path/to/a.wav"][0]["start"] == 10.0
    
    assert len(unstitched["/path/to/b.wav"]) == 1
    assert unstitched["/path/to/b.wav"][0]["text"] == "World"
    assert unstitched["/path/to/b.wav"][0]["start"] == 100.0 # 3700 - 3600
